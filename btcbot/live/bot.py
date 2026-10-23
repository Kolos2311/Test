"""One strategy = one bot = one position at a time on one account.

Each poll:
  1. kill switch: a file named STOP in the bot's state dir closes the position,
     cancels orders and stops the bot;
  2. sync with the exchange: detect a closed trade (log P&L, update risk
     counters, cancel leftover orders), adopt an unknown open position, re-place
     missing TP/SL orders;
  3. when flat, right after a timeframe bar closes: evaluate the strategy on that
     bar and, on a signal, enter at market and place reduce-only TP and SL.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .broker import OrderError, Position
from .config import HTF_FOR, BotConfig
from .market import MarketData, now_ms
from .signals import evaluate

ENTRY_DELAY_S = 3        # wait for the exchange to finalize the closed candle
MAX_ENTRY_LATENESS_S = 60  # backtest enters at the next minute open; don't chase later
CLOSE_FILL_WAIT_POLLS = 6


@dataclass
class BotState:
    last_bar: str | None = None
    trade: dict | None = None          # the open trade as the bot knows it
    day: str = ""
    day_pnl: float = 0.0
    consecutive_losses: int = 0
    paused_reason: str | None = None
    stats: dict = field(default_factory=lambda: {"trades": 0, "wins": 0, "pnl": 0.0})


class Bot:
    def __init__(self, cfg: BotConfig, broker, market: MarketData, clock=now_ms):
        self.clock = clock
        self.cfg = cfg
        self.broker = broker
        self.market = market
        self.log = logging.getLogger(f"bot.{cfg.name}")
        self.dir = cfg.state_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self.stopped = False

    # ---------- persistence ----------
    def _load(self) -> BotState:
        f = self.dir / "state.json"
        return BotState(**json.loads(f.read_text())) if f.exists() else BotState()

    def _save(self) -> None:
        (self.dir / "state.json").write_text(json.dumps(self.state.__dict__, indent=1, default=str))

    def _journal(self, row: dict) -> None:
        f = self.dir / "trades.csv"
        new = not f.exists()
        with f.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def reset_risk(self) -> None:
        self.state.consecutive_losses = 0
        self.state.paused_reason = None
        self._save()

    # ---------- main step ----------
    def step(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if (self.dir / "STOP").exists():
            self.log.warning("STOP file found: closing position and stopping")
            self.broker.cancel_all()
            if self.broker.position():
                self.broker.close()
            self.stopped = True
            return

        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        if day != self.state.day:
            self.state.day, self.state.day_pnl = day, 0.0
            if self.state.paused_reason == "daily loss limit":
                self.state.paused_reason = None
            self._save()

        pos = self.broker.position()
        self._sync(pos, now)
        if pos is None:
            self._maybe_enter(now)

    def _sync(self, pos: Position | None, now: float) -> None:
        tr = self.state.trade
        if tr and pos is None:
            self._on_closed(tr)
            return
        if pos is None:
            return
        if tr is None:
            # position we did not open in this session (restart / manual): manage it
            tp, sl = self._levels(pos.is_long, pos.entry_px)
            self.log.warning("adopting existing position %s, TP %.1f SL %.1f", pos, tp, sl)
            tr = self.state.trade = {"side": "long" if pos.is_long else "short", "entry_px": pos.entry_px,
                                     "size": abs(pos.size), "tp": tp, "sl": sl, "oids": [],
                                     "opened_ms": self.clock(), "bar": None}
            self._save()
        if not set(tr.get("oids", [])) or not set(tr["oids"]) <= self.broker.open_order_ids():
            self.log.warning("TP/SL missing, placing them again")
            self.broker.cancel_all()
            tr["oids"] = self.broker.protect(pos, tr["tp"], tr["sl"])
            self._save()

    def _on_closed(self, tr: dict) -> None:
        self.broker.cancel_all()
        pnl, fees, exit_px = self.broker.closed_pnl_since(tr["opened_ms"])
        if exit_px is None and tr.get("close_wait", 0) < CLOSE_FILL_WAIT_POLLS:
            # the closing fill is not in the exchange history yet: look again next poll
            tr["close_wait"] = tr.get("close_wait", 0) + 1
            self._save()
            return
        net = pnl - fees
        win = pnl > 0
        st = self.state
        st.trade = None
        st.day_pnl += net
        st.consecutive_losses = 0 if win else st.consecutive_losses + 1
        st.stats["trades"] += 1
        st.stats["wins"] += int(win)
        st.stats["pnl"] += net
        self._journal({
            "bot": self.cfg.name, "mode": self.cfg.mode, "bar": tr.get("bar"), "side": tr["side"],
            "opened": pd.Timestamp(tr["opened_ms"], unit="ms", tz="UTC").isoformat(),
            "closed": pd.Timestamp.now(tz="UTC").isoformat(), "entry_px": tr["entry_px"],
            "exit_px": exit_px, "size": tr["size"], "tp": tr["tp"], "sl": tr["sl"],
            "result": "win" if win else "loss", "pnl_usd": round(pnl, 4), "fees_usd": round(fees, 4),
            "net_usd": round(net, 4)})
        self.log.info("trade closed: %s net %.2f USD (day %.2f, streak of losses %d, total %s)",
                      "WIN" if win else "LOSS", net, st.day_pnl, st.consecutive_losses, st.stats)
        if st.day_pnl <= -self.cfg.max_daily_loss_usd:
            st.paused_reason = "daily loss limit"
        if st.consecutive_losses >= self.cfg.max_consecutive_losses:
            st.paused_reason = "consecutive losses limit (restart with --reset-risk)"
        if st.paused_reason:
            self.log.warning("paused: %s", st.paused_reason)
        self._save()

    def _levels(self, is_long: bool, px: float) -> tuple[float, float]:
        tp, sl = self.cfg.take_profit_pct / 100, self.cfg.stop_loss_pct / 100
        return (px * (1 + tp), px * (1 - sl)) if is_long else (px * (1 - tp), px * (1 + sl))

    def _maybe_enter(self, now: float) -> None:
        step = pd.Timedelta(self.cfg.timeframe)
        bar_close = pd.Timestamp(now, unit="s", tz="UTC").floor(step)
        if self.state.last_bar and pd.Timestamp(self.state.last_bar) >= bar_close:
            return
        lateness = now - bar_close.timestamp()
        if lateness < ENTRY_DELAY_S:
            return
        if lateness > MAX_ENTRY_LATENESS_S:
            self.log.info("bar %s: too late to act (%.0fs after close), waiting for the next bar",
                          bar_close, lateness)
            self.state.last_bar = bar_close.isoformat()
            self._save()
            return

        frames = self.market.frames(self.cfg.timeframe, HTF_FOR[self.cfg.timeframe])
        sig = evaluate(self.cfg.timeframe, self.cfg.conditions, frames)
        if sig.bar_close != bar_close:
            self.log.info("bar %s: exchange still shows %s as the last closed candle, retrying",
                          bar_close, sig.bar_close)
            return
        self.state.last_bar = bar_close.isoformat()
        self._save()
        self.log.info("bar %s: long=%s short=%s", bar_close, sig.long, sig.short)
        if sig.side is None:
            return
        if self.state.paused_reason:
            self.log.warning("signal %s ignored: paused (%s)", sig.side, self.state.paused_reason)
            return

        is_long = sig.side == "long"
        size = self.cfg.notional_usd / self.broker.mid()
        opened_ms = self.clock() - 1000       # fills of this trade are looked up from here
        try:
            fill = self.broker.open(is_long, size)
        except OrderError as e:
            self.log.error("entry failed: %s", e)
            return
        pos = self.broker.position()
        if pos is None:
            self.log.error("entry reported %s but no position found", fill)
            return
        tp, sl = self._levels(is_long, fill.avg_px)
        self.state.trade = {"side": sig.side, "entry_px": fill.avg_px, "size": abs(pos.size),
                            "tp": tp, "sl": sl, "oids": [], "opened_ms": opened_ms,
                            "bar": bar_close.isoformat()}
        self._save()
        self.log.info("ENTER %s %.5f @ %.1f  TP %.1f  SL %.1f", sig.side, abs(pos.size), fill.avg_px, tp, sl)
        try:
            self.state.trade["oids"] = self.broker.protect(pos, tp, sl)
        except OrderError as e:
            # never sit in a position without a stop
            self.log.error("could not place TP/SL (%s): closing the position", e)
            self.broker.close()
        self._save()

    def run(self, poll_s: float = 15.0) -> None:
        self.log.info("started: %s %s %s, conditions %s", self.cfg.mode, self.cfg.coin, self.cfg.timeframe,
                      self.cfg.conditions)
        errors = 0
        while not self.stopped:
            try:
                self.step()
                errors = 0
            except Exception:  # network hiccups etc.: keep the bot alive, back off
                errors += 1
                self.log.exception("step failed (%d in a row)", errors)
                time.sleep(min(60, 5 * errors))
            time.sleep(poll_s)
