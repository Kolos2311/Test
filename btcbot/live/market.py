"""Public Hyperliquid market data: candles and mid prices.

Candles are returned in the same layout the research uses: indexed by bar
*close* time (UTC), only fully closed bars, Hyperliquid volume in BTC.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import requests

MAINNET_API_URL = "https://api.hyperliquid.xyz"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
HL_INTERVAL = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h", "4h": "4h", "1D": "1d"}
MAX_CANDLES = 5000   # Hyperliquid serves at most the latest 5000 candles per interval


def now_ms() -> int:
    return int(time.time() * 1000)


class PublicClient:
    def __init__(self, base_url: str = MAINNET_API_URL, timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

    def post(self, payload: dict):
        resp = self.session.post(f"{self.base_url}/info", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


BASE_BARS = 1500   # enough for EMA200 to converge and the 500-bar rolling ranks
HTF_BARS = 300


class MarketData:
    """Candle requests are cached per timeframe until the next bar closes, so bots
    on the same timeframe share them and stay well inside Hyperliquid's rate limits."""

    def __init__(self, client: PublicClient, coin: str = "BTC"):
        self.client = client
        self.coin = coin
        self._cache: dict = {}
        self._lock = threading.Lock()

    def mid(self) -> float:
        return float(self.client.post({"type": "allMids"})[self.coin])

    def candles(self, tf: str, bars: int = MAX_CANDLES, end_ms: int | None = None) -> pd.DataFrame:
        """Closed candles of timeframe `tf` up to `end_ms` (default: now)."""
        end_ms = end_ms or now_ms()
        step = int(pd.Timedelta(tf).total_seconds() * 1000)
        start = end_ms - min(bars, MAX_CANDLES) * step
        raw = self.client.post({"type": "candleSnapshot", "req": {
            "coin": self.coin, "interval": HL_INTERVAL[tf], "startTime": start, "endTime": end_ms}})
        return to_frame(raw, step, end_ms)

    def cached_candles(self, tf: str, bars: int) -> pd.DataFrame:
        """Candles shared between bots. Only a response that already contains the
        latest closed bar is cached, so a new bar is never hidden by the cache."""
        expected = pd.Timestamp.now(tz="UTC").floor(pd.Timedelta(tf))
        key = (tf, bars)
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == expected:
                return hit[1]
        df = self.candles(tf, bars)
        if len(df) and df.index[-1] >= expected:
            with self._lock:
                self._cache[key] = (expected, df)
        return df

    def frames(self, base_tf: str, htfs: list[str]) -> dict[str, pd.DataFrame]:
        out = {base_tf: self.cached_candles(base_tf, BASE_BARS)}
        out.update({tf: self.cached_candles(tf, HTF_BARS) for tf in htfs})
        return out


def to_frame(raw: list[dict], step_ms: int, end_ms: int) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume",
                                     "count", "taker_buy_volume"])
    t = np.array([int(c["t"]) for c in raw], dtype=np.int64)
    close_t = t + step_ms
    df = pd.DataFrame({
        "open": [float(c["o"]) for c in raw],
        "high": [float(c["h"]) for c in raw],
        "low": [float(c["l"]) for c in raw],
        "close": [float(c["c"]) for c in raw],
        "volume": [float(c["v"]) for c in raw],
        "count": [float(c.get("n", np.nan)) for c in raw],
    }, index=pd.to_datetime(close_t, unit="ms", utc=True))
    df.index.name = "close_time"
    df["quote_volume"] = df["volume"] * df["close"]
    df["taker_buy_volume"] = np.nan          # not provided by Hyperliquid candles
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # the candle still forming has close time in the future: drop it
    return df[df.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
