"""Order execution: real Hyperliquid (testnet / mainnet) and a paper broker.

Both expose the same small interface used by the bot:
    position()                -> Position | None
    open(is_long, size)       -> Fill            (IOC market-style entry)
    protect(pos, tp, sl)      -> list[int]       (reduce-only TP + SL trigger orders, returns oids)
    open_order_ids()          -> set[int]
    cancel_all()
    close()                                       (reduce-only market close)
    closed_pnl_since(ms)      -> (pnl_usd, fees_usd, exit_px)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .market import MarketData, now_ms

log = logging.getLogger(__name__)

TAKER_FEE = 0.00045   # Hyperliquid base tier
TRIGGER_SLIPPAGE = 0.01  # limit price allowance of trigger-market TP/SL orders


class OrderError(RuntimeError):
    pass


@dataclass
class Position:
    size: float        # signed: > 0 long, < 0 short
    entry_px: float

    @property
    def is_long(self) -> bool:
        return self.size > 0


@dataclass
class Fill:
    size: float
    avg_px: float
    oid: int | None = None


def round_px(px: float, sz_decimals: int) -> float:
    """Hyperliquid perp prices: at most 5 significant figures and 6 - szDecimals decimals."""
    return round(float(f"{px:.5g}"), 6 - sz_decimals)


def round_size(sz: float, sz_decimals: int) -> float:
    q = 10 ** sz_decimals
    return math.floor(sz * q + 1e-9) / q


def _statuses(resp) -> list[dict]:
    if not isinstance(resp, dict) or resp.get("status") != "ok":
        raise OrderError(f"exchange rejected request: {resp}")
    data = resp.get("response", {}).get("data", {})
    return data.get("statuses", [])


class HyperliquidBroker:
    def __init__(self, coin: str, exchange, address: str, entry_slippage: float):
        """`exchange` is a hyperliquid.exchange.Exchange; `address` is the account
        (or sub-account) whose position this bot manages."""
        self.coin = coin
        self.exchange = exchange
        self.info = exchange.info
        self.address = address
        self.entry_slippage = entry_slippage
        self.sz_decimals = self.info.asset_to_sz_decimals[self.info.name_to_asset(coin)]

    @classmethod
    def connect(cls, coin: str, base_url: str, secret_key: str, account_address: str,
                subaccount: str, entry_slippage: float) -> "HyperliquidBroker":
        import eth_account
        from hyperliquid.exchange import Exchange

        wallet = eth_account.Account.from_key(secret_key)
        exchange = Exchange(wallet, base_url, account_address=account_address,
                            vault_address=subaccount or None)
        return cls(coin, exchange, subaccount or account_address, entry_slippage)

    def setup(self, leverage: int, isolated: bool) -> None:
        resp = self.exchange.update_leverage(leverage, self.coin, not isolated)
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            raise OrderError(f"update_leverage failed: {resp}")

    def mid(self) -> float:
        return float(self.info.all_mids()[self.coin])

    def equity(self) -> float:
        return float(self.info.user_state(self.address)["marginSummary"]["accountValue"])

    def position(self) -> Position | None:
        for p in self.info.user_state(self.address)["assetPositions"]:
            item = p["position"]
            if item["coin"] == self.coin and float(item["szi"]) != 0:
                return Position(float(item["szi"]), float(item["entryPx"]))
        return None

    def open(self, is_long: bool, size: float) -> Fill:
        mid = self.mid()
        px = round_px(mid * (1 + self.entry_slippage if is_long else 1 - self.entry_slippage), self.sz_decimals)
        sz = round_size(size, self.sz_decimals)
        resp = self.exchange.order(self.coin, is_long, sz, px, {"limit": {"tif": "Ioc"}}, reduce_only=False)
        st = _statuses(resp)
        if not st or "filled" not in st[0]:
            raise OrderError(f"entry not filled: {resp}")
        f = st[0]["filled"]
        return Fill(float(f["totalSz"]), float(f["avgPx"]), f.get("oid"))

    def protect(self, pos: Position, tp_px: float, sl_px: float) -> list[int]:
        close_buy = not pos.is_long
        sz = round_size(abs(pos.size), self.sz_decimals)

        def trigger(px, kind):
            limit = px * (1 + TRIGGER_SLIPPAGE if close_buy else 1 - TRIGGER_SLIPPAGE)
            return {"coin": self.coin, "is_buy": close_buy, "sz": sz,
                    "limit_px": round_px(limit, self.sz_decimals),
                    "order_type": {"trigger": {"triggerPx": round_px(px, self.sz_decimals),
                                               "isMarket": True, "tpsl": kind}},
                    "reduce_only": True}

        resp = self.exchange.bulk_orders([trigger(tp_px, "tp"), trigger(sl_px, "sl")])
        oids = []
        for s in _statuses(resp):
            if "resting" in s:
                oids.append(int(s["resting"]["oid"]))
            elif "filled" in s:          # level already crossed: closed immediately
                oids.append(int(s["filled"]["oid"]))
            else:
                raise OrderError(f"TP/SL order failed: {resp}")
        return oids

    def open_order_ids(self) -> set[int]:
        return {int(o["oid"]) for o in self.info.frontend_open_orders(self.address) if o["coin"] == self.coin}

    def cancel_all(self) -> None:
        for o in self.info.frontend_open_orders(self.address):
            if o["coin"] == self.coin:
                self.exchange.cancel(self.coin, int(o["oid"]))

    def close(self) -> None:
        self.exchange.market_close(self.coin, slippage=0.01)

    def closed_pnl_since(self, start_ms: int) -> tuple[float, float, float | None]:
        fills = [f for f in self.info.user_fills_by_time(self.address, start_ms) if f["coin"] == self.coin]
        pnl = sum(float(f.get("closedPnl", 0)) for f in fills)
        fees = sum(float(f.get("fee", 0)) for f in fills)
        closing = [f for f in fills if f.get("dir", "").startswith("Close")]
        exit_px = float(closing[-1]["px"]) if closing else None
        return pnl, fees, exit_px


class PaperBroker:
    """Simulated execution on live Hyperliquid prices with the backtest rules:
    fill at mid, TP/SL checked on closed 1m candles, both touched in one candle
    = stop-loss. Taker fee on entry and exit."""

    def __init__(self, coin: str, market: MarketData, state_file: Path, clock=now_ms):
        self.coin = coin
        self.market = market
        self.state_file = state_file
        self.clock = clock
        self.sz_decimals = 5
        self.pos: dict | None = None
        self.fills: list[dict] = []
        if state_file.exists():
            s = json.loads(state_file.read_text())
            self.pos, self.fills = s.get("pos"), s.get("fills", [])

    def _save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"pos": self.pos, "fills": self.fills[-500:]}, indent=1))

    def setup(self, leverage: int, isolated: bool) -> None:
        pass

    def mid(self) -> float:
        return self.market.mid()

    def equity(self) -> float:
        return float("nan")

    def position(self) -> Position | None:
        self._check_exit()
        return Position(self.pos["size"], self.pos["entry_px"]) if self.pos else None

    def open(self, is_long: bool, size: float) -> Fill:
        px = self.mid()
        sz = round_size(size, self.sz_decimals)
        t = self.clock()
        self.pos = {"size": sz if is_long else -sz, "entry_px": px, "opened_ms": t,
                    "checked_ms": t, "tp": None, "sl": None}
        self.fills.append({"time": t, "kind": "open", "px": px, "sz": sz, "closedPnl": 0.0,
                           "fee": px * sz * TAKER_FEE})
        self._save()
        return Fill(sz, px)

    def protect(self, pos: Position, tp_px: float, sl_px: float) -> list[int]:
        self.pos["tp"], self.pos["sl"] = tp_px, sl_px
        self._save()
        return [1, 2]

    def open_order_ids(self) -> set[int]:
        return {1, 2} if self.pos and self.pos["tp"] is not None else set()

    def cancel_all(self) -> None:
        if self.pos:
            self.pos["tp"] = self.pos["sl"] = None
            self._save()

    def close(self) -> None:
        if self.pos:
            self._exit(self.mid(), self.clock())

    def closed_pnl_since(self, start_ms: int) -> tuple[float, float, float | None]:
        fills = [f for f in self.fills if f["time"] >= start_ms]
        closing = [f for f in fills if f["kind"] == "close"]
        return (sum(f["closedPnl"] for f in fills), sum(f["fee"] for f in fills),
                closing[-1]["px"] if closing else None)

    def _exit(self, px: float, t: int) -> None:
        sz, entry = self.pos["size"], self.pos["entry_px"]
        self.fills.append({"time": t, "kind": "close", "px": px, "sz": abs(sz),
                           "closedPnl": (px - entry) * sz, "fee": px * abs(sz) * TAKER_FEE})
        self.pos = None
        self._save()

    def _check_exit(self) -> None:
        if not self.pos or self.pos["tp"] is None:
            return
        checked = self.pos["checked_ms"]
        if self.clock() - checked < 61_000:      # no new closed 1m candle yet
            return
        bars = self.market.candles("1min", bars=(self.clock() - checked) // 60_000 + 2, end_ms=self.clock())
        # closed 1m candles not inspected yet; the first one contains the entry
        bars = bars[bars.index > pd.Timestamp(checked, unit="ms", tz="UTC")]
        long = self.pos["size"] > 0
        tp, sl = self.pos["tp"], self.pos["sl"]
        for t, b in bars.iterrows():
            hit_sl = b["low"] <= sl if long else b["high"] >= sl
            hit_tp = b["high"] >= tp if long else b["low"] <= tp
            if hit_sl or hit_tp:
                self._exit(sl if hit_sl else tp, int(t.timestamp() * 1000))
                log.info("paper exit %s at %s", "SL" if hit_sl else "TP", t)
                return
        if len(bars):
            self.pos["checked_ms"] = int(bars.index[-1].timestamp() * 1000)
            self._save()

    def snapshot(self) -> dict:
        return {"pos": self.pos, "fills": len(self.fills)}


def describe(pos: Position | None) -> str:
    return "flat" if pos is None else json.dumps(asdict(pos))
