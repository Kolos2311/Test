import numpy as np
import pandas as pd
import pytest

from btcbot import conditions, data
from btcbot.live import bot as botmod
from btcbot.live.bot import Bot
from btcbot.live.broker import HyperliquidBroker, OrderError, PaperBroker, Position, round_px, round_size
from btcbot.live.config import HTF_FOR, BotConfig
from btcbot.live.market import BASE_BARS, HTF_BARS, to_frame
from btcbot.live.signals import Signal, evaluate
from tests.test_conditions import synthetic_1m


def test_rounding_follows_hyperliquid_rules():
    assert round_px(65432.17, 5) == 65432.0
    assert round_px(101234.5, 5) == 101230.0          # 5 significant figures
    assert round_px(9876.543, 5) == 9876.5             # at most 1 decimal for szDecimals=5
    assert round_size(0.000159, 5) == 0.00015          # never round size up


def test_to_frame_drops_forming_candle():
    step = 15 * 60_000
    raw = [{"t": i * step, "T": (i + 1) * step - 1, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "3", "n": 7}
           for i in range(4)]
    df = to_frame(raw, step, end_ms=3 * step + 10)       # 4th candle closes at 4*step: still open
    assert len(df) == 3
    assert df.index[-1] == pd.Timestamp(3 * step, unit="ms", tz="UTC")


# ---------------- Hyperliquid broker against a fake SDK ----------------

class FakeInfo:
    asset_to_sz_decimals = {0: 5}

    def __init__(self):
        self.positions = []
        self.orders = []
        self.fills = []

    def name_to_asset(self, name):
        return 0

    def all_mids(self):
        return {"BTC": "60000"}

    def user_state(self, addr):
        return {"assetPositions": [{"position": p} for p in self.positions],
                "marginSummary": {"accountValue": "100"}}

    def frontend_open_orders(self, addr):
        return self.orders

    def user_fills_by_time(self, addr, start):
        return self.fills


class FakeExchange:
    def __init__(self, info):
        self.info = info
        self.calls = []
        self.next_status = None

    def ok(self, statuses):
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": statuses}}}

    def order(self, coin, is_buy, sz, px, order_type, reduce_only=False):
        self.calls.append(("order", coin, is_buy, sz, px, order_type, reduce_only))
        st = self.next_status or {"filled": {"totalSz": str(sz), "avgPx": "60010.0", "oid": 11}}
        if "filled" in st:
            self.info.positions = [{"coin": coin, "szi": str(sz if is_buy else -sz), "entryPx": "60010.0"}]
        return self.ok([st])

    def bulk_orders(self, reqs, builder=None, grouping="na"):
        self.calls.append(("bulk", reqs))
        oids = [100 + i for i in range(len(reqs))]
        self.info.orders = [{"coin": r["coin"], "oid": o} for r, o in zip(reqs, oids)]
        return self.ok([{"resting": {"oid": o}} for o in oids])

    def cancel(self, coin, oid):
        self.calls.append(("cancel", oid))
        self.info.orders = [o for o in self.info.orders if o["oid"] != oid]

    def market_close(self, coin, slippage=0.05):
        self.calls.append(("close",))
        self.info.positions = []

    def update_leverage(self, lev, coin, is_cross):
        self.calls.append(("leverage", lev, is_cross))
        return {"status": "ok"}


def make_hl():
    info = FakeInfo()
    ex = FakeExchange(info)
    return HyperliquidBroker("BTC", ex, "0xsub", entry_slippage=0.003), ex, info


def test_entry_is_ioc_with_slippage_cap():
    br, ex, _ = make_hl()
    fill = br.open(True, 0.0012345)
    _, coin, is_buy, sz, px, otype, reduce_only = ex.calls[0]
    assert (coin, is_buy, sz, reduce_only) == ("BTC", True, 0.00123, False)
    assert px == 60180.0 and otype == {"limit": {"tif": "Ioc"}}
    assert fill.avg_px == 60010.0 and fill.size == 0.00123


def test_entry_rejection_raises():
    br, ex, _ = make_hl()
    ex.next_status = {"error": "Insufficient margin"}
    with pytest.raises(OrderError):
        br.open(True, 0.001)


def test_protect_places_reduce_only_trigger_orders():
    br, ex, _ = make_hl()
    oids = br.protect(Position(0.001, 60000.0), tp_px=60600.0, sl_px=59850.0)
    reqs = ex.calls[-1][1]
    tp, sl = reqs
    assert oids == [100, 101]
    assert all(r["reduce_only"] and r["is_buy"] is False and r["sz"] == 0.001 for r in reqs)
    assert tp["order_type"]["trigger"] == {"triggerPx": 60600.0, "isMarket": True, "tpsl": "tp"}
    assert sl["order_type"]["trigger"] == {"triggerPx": 59850.0, "isMarket": True, "tpsl": "sl"}
    assert sl["limit_px"] < 59850.0                    # sell stop may fill below the trigger
    short = br.protect(Position(-0.001, 60000.0), tp_px=59400.0, sl_px=60150.0)
    assert short and all(r["is_buy"] for r in ex.calls[-1][1])


# ---------------- bot state machine ----------------

class FakeMarket:
    def __init__(self):
        self.price = 60000.0
        self.minute_bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def mid(self):
        return self.price

    def frames(self, tf, htfs):
        return {}

    def candles(self, tf, bars=0, end_ms=None):
        assert tf == "1min"
        b = self.minute_bars
        return b[b.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")] if end_ms else b


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return int(self.t * 1000)


def make_bot(tmp_path, monkeypatch, signal_side="long", broker="paper", **kw):
    cfg = BotConfig(name="t", timeframe="15min", conditions=["x"], state_dir=str(tmp_path),
                    notional_usd=60.0, **kw)
    t0 = pd.Timestamp("2026-10-23 12:15", tz="UTC").timestamp()
    clock = Clock(t0 + 5)
    market = FakeMarket()

    def fake_eval(tf, names, frames):
        bar = pd.Timestamp(clock.t, unit="s", tz="UTC").floor("15min")
        return Signal(bar, signal_side == "long", signal_side == "short")

    monkeypatch.setattr(botmod, "evaluate", fake_eval)
    if broker == "paper":
        br = PaperBroker("BTC", market, tmp_path / "paper.json", clock=clock)
    else:
        br, _, _ = make_hl()
    return Bot(cfg, br, market, clock=clock), market, clock


def test_paper_trade_full_cycle(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch)
    bot.step(clock.t)
    tr = bot.state.trade
    assert tr["side"] == "long" and tr["entry_px"] == 60000.0
    assert tr["tp"] == pytest.approx(60600.0) and tr["sl"] == pytest.approx(59850.0)
    assert bot.broker.position().size == 0.001

    # same bar again: no second entry
    bot.step(clock.t + 20)
    assert len([f for f in bot.broker.fills if f["kind"] == "open"]) == 1

    # next minute candle touches the stop
    entry_min = pd.Timestamp("2026-10-23 12:16", tz="UTC")
    market.minute_bars = pd.DataFrame({"open": [60000.0], "high": [60100.0], "low": [59800.0],
                                       "close": [59900.0], "volume": [1.0]}, index=[entry_min])
    clock.t += 70
    bot.step(clock.t)
    assert bot.state.trade is None
    assert bot.state.consecutive_losses == 1
    rows = (tmp_path / "t" / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",loss," in rows[1]


def test_both_levels_in_one_candle_is_a_loss(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch, signal_side="short")
    bot.step(clock.t)
    market.minute_bars = pd.DataFrame({"open": [60000.0], "high": [60200.0], "low": [59300.0],
                                       "close": [59500.0], "volume": [1.0]},
                                      index=[pd.Timestamp("2026-10-23 12:16", tz="UTC")])
    clock.t += 70
    bot.step(clock.t)
    assert bot.state.stats == {"trades": 1, "wins": 0, "pnl": pytest.approx(-0.15 - 0.054, abs=0.01)}


def test_daily_loss_limit_pauses_entries(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch, max_daily_loss_usd=0.1)
    bot.step(clock.t)
    market.minute_bars = pd.DataFrame({"open": [60000.0], "high": [60000.0], "low": [59000.0],
                                       "close": [59100.0], "volume": [1.0]},
                                      index=[pd.Timestamp("2026-10-23 12:16", tz="UTC")])
    clock.t += 70
    bot.step(clock.t)
    assert bot.state.paused_reason == "daily loss limit"
    clock.t += 15 * 60                                   # next bar has a signal too
    bot.step(clock.t)
    assert bot.state.trade is None


def test_late_start_skips_the_bar(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch)
    clock.t += 120                                       # 2 minutes after the close
    bot.step(clock.t)
    assert bot.state.trade is None


def test_stop_file_closes_position(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch)
    bot.step(clock.t)
    (tmp_path / "t" / "STOP").write_text("")
    bot.step(clock.t + 10)
    assert bot.stopped and bot.broker.position() is None


def test_live_bot_replaces_missing_stop(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch, broker="hl")
    bot.step(clock.t)
    ex = bot.broker.exchange
    assert ex.calls[0][0] == "order" and ex.calls[1][0] == "bulk"
    bot.broker.info.orders = bot.broker.info.orders[:1]  # SL order vanished
    bot.step(clock.t + 10)
    assert ex.calls[-1][0] == "bulk" and len(bot.broker.info.orders) == 2


def test_live_bot_adopts_unknown_position(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch, broker="hl", signal_side=None)
    bot.broker.info.positions = [{"coin": "BTC", "szi": "-0.002", "entryPx": "61000"}]
    bot.step(clock.t)
    tr = bot.state.trade
    assert tr["side"] == "short" and tr["sl"] == pytest.approx(61000 * 1.0025)
    assert len(bot.broker.info.orders) == 2


# ---------------- live signals == backtest signals ----------------

@pytest.fixture(scope="module")
def long_synthetic():
    return synthetic_1m(n=60 * 24 * 120, seed=5)


@pytest.mark.parametrize("tf", ["5min", "15min", "1h"])
def test_live_window_signals_match_backtest(tf, long_synthetic):
    """Signals computed by the bot from the trailing BASE_BARS / HTF_BARS candles
    must equal the backtest's full-history conditions on the same bar."""
    df = long_synthetic
    full_base = data.resample(df, tf)
    full_htf = {h: data.resample(df, h) for h in HTF_FOR[tf]}
    full = conditions.build(full_base, full_htf)
    names = [c.name for c in full]
    mismatches = {}
    for i in range(len(full_base) - 40, len(full_base) - 1, 3):
        t = full_base.index[i]
        frames = {tf: full_base[full_base.index <= t].tail(BASE_BARS)}
        frames.update({h: f[f.index <= t].tail(HTF_BARS) for h, f in full_htf.items()})
        live = {c.name: c for c in conditions.build(frames[tf], {h: frames[h] for h in HTF_FOR[tf]})}
        assert list(live) == names
        for c in full:
            if bool(c.long[i]) != bool(live[c.name].long[-1]) or bool(c.short[i]) != bool(live[c.name].short[-1]):
                mismatches[c.name] = mismatches.get(c.name, 0) + 1
        sig = evaluate(tf, names[:2], frames)
        assert sig.bar_close == t
    assert not mismatches, mismatches


def test_config_load_and_account_conflicts(tmp_path, monkeypatch):
    from btcbot.live import config as cfgmod
    from btcbot.live.__main__ import check_accounts

    p = tmp_path / "b.toml"
    p.write_text('name = "b1"\ntimeframe = "15min"\nconditions = ["RSI2<10"]\nmode = "testnet"\n'
                 '[backtest]\ntrain_trades = 10\n')
    a = cfgmod.load(p)
    assert a.take_profit_pct == 1.0 and a.stop_loss_pct == 0.25 and a.backtest == {"train_trades": 10}
    b = cfgmod.BotConfig(name="b2", timeframe="1h", conditions=["x"], mode="testnet")
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0xMAIN")
    with pytest.raises(SystemExit):
        check_accounts([a, b])                        # same main account, same coin
    b.subaccount_address = "0xSUB"
    check_accounts([a, b])

    bad = cfgmod.BotConfig(name="m", timeframe="1h", conditions=["x"], mode="mainnet")
    monkeypatch.delenv("HL_CONFIRM_MAINNET", raising=False)
    with pytest.raises(ValueError):
        bad.validate()
    monkeypatch.setenv("HL_CONFIRM_MAINNET", "yes")
    bad.validate()


def test_shipped_bot_configs_are_valid(long_synthetic):
    from pathlib import Path

    from btcbot.live import config as cfgmod
    from btcbot.live.signals import available_conditions

    paths = sorted(Path(__file__).resolve().parent.parent.joinpath("bots").glob("*.toml"))
    assert len(paths) == 5
    for p in paths:
        c = cfgmod.load(p)
        assert c.mode == "paper" and c.take_profit_pct == 1.0 and c.stop_loss_pct == 0.25
        frames = {tf: data.resample(long_synthetic, tf) for tf in [c.timeframe, *HTF_FOR[c.timeframe]]}
        assert set(c.conditions) <= set(available_conditions(c.timeframe, frames)), p.name


def test_live_close_waits_for_fill_history(tmp_path, monkeypatch):
    bot, market, clock = make_bot(tmp_path, monkeypatch, broker="hl")
    bot.step(clock.t)
    info = bot.broker.info
    info.positions, info.orders = [], [o for o in info.orders]   # TP hit: position gone
    bot.step(clock.t + 15)                                         # fills not indexed yet
    assert bot.state.trade is not None and bot.state.trade["close_wait"] == 1
    info.fills = [{"coin": "BTC", "dir": "Close Long", "px": "60610", "closedPnl": "0.6", "fee": "0.03"}]
    bot.step(clock.t + 30)
    assert bot.state.trade is None and bot.state.stats["wins"] == 1
    assert info.orders == []                                       # leftover SL cancelled
