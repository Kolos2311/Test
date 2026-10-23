"""Historical 1m klines for Binance USDT-M futures from data.binance.vision.

Monthly archives cover full months; the current month is filled from daily
archives. Everything is stored as one parquet file indexed by open time (UTC).
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data/futures/um"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch(url: str, retries: int = 4) -> bytes | None:
    """Download url; None when the archive does not exist (404)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {err}")


def _parse_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(zf.namelist()[0])
    # Newer archives carry a header row, older ones do not.
    has_header = not raw[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = COLUMNS
    ts = df["open_time"].astype("int64")
    # Some archives switched to microsecond timestamps.
    ts = np.where(ts > 10**14, ts // 1000, ts)
    df.index = pd.to_datetime(ts, unit="ms", utc=True)
    df.index.name = "open_time"
    return df[KEEP].astype("float64")


def download(symbol: str = "BTCUSDT", interval: str = "1m",
             start: str = "2020-01", end: date | None = None,
             cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Download klines from `start` (YYYY-MM) through `end` (default: yesterday)."""
    end = end or (date.today() - timedelta(days=1))
    raw_dir = cache_dir / "raw" / symbol / interval
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    month = pd.Period(start, freq="M")
    last_full_month = pd.Period(end, freq="M") - 1
    while month <= last_full_month:
        name = f"{symbol}-{interval}-{month.strftime('%Y-%m')}.zip"
        frames.append(_load_cached(f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{name}",
                                   raw_dir / name))
        month += 1
    day = pd.Period(end, freq="M").start_time.date()
    while day <= end:
        name = f"{symbol}-{interval}-{day.isoformat()}.zip"
        frames.append(_load_cached(f"{BASE_URL}/daily/klines/{symbol}/{interval}/{name}",
                                   raw_dir / name))
        day += timedelta(days=1)

    df = pd.concat([f for f in frames if f is not None])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _load_cached(url: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        blob = _fetch(url)
        if blob is None:
            print(f"  missing archive: {url}")
            return None
        path.write_bytes(blob)
        print(f"  downloaded {path.name}")
    return _parse_zip(path.read_bytes())


def save(df: pd.DataFrame, symbol: str = "BTCUSDT", interval: str = "1m",
         cache_dir: Path = DATA_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{symbol}_{interval}.parquet"
    df.to_parquet(out)
    return out


def load(symbol: str = "BTCUSDT", interval: str = "1m", cache_dir: Path | None = None) -> pd.DataFrame:
    path = (cache_dir or DATA_DIR) / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m research.fetch_data` first")
    return pd.read_parquet(path)


def load_bitstamp(repo: Path, start: str = "2019-10-01") -> pd.DataFrame:
    """Bitstamp BTC/USD 1m candles from a clone of github.com/ff137/bitstamp-btcusd-minute-data
    (bulk history + daily updates). Bitstamp has no taker-side volume, so that
    column is NaN and order-flow conditions are skipped."""
    dt = {"timestamp": "int64", "open": "float64", "high": "float64",
          "low": "float64", "close": "float64", "volume": "float64"}
    parts = [pd.read_csv(p, dtype=dt) for p in sorted((repo / "data" / "historical").glob("*.csv.gz"))]
    parts += [pd.read_csv(p, dtype=dt) for p in sorted((repo / "data" / "updates").glob("*.csv"))]
    df = pd.concat(parts).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    df.index = pd.to_datetime(df.pop("timestamp"), unit="s", utc=True)
    df.index.name = "open_time"
    df = df[df.index >= pd.Timestamp(start, tz="UTC")].copy()
    df["quote_volume"] = df["volume"] * df["close"]
    df["count"] = np.nan
    df["taker_buy_volume"] = np.nan
    return clip_isolated_wicks(df)


def clip_isolated_wicks(df: pd.DataFrame, limit: float = 0.015) -> pd.DataFrame:
    """Cap single-minute spikes that stick out more than `limit` beyond both the
    previous and the next close (bad ticks on a thin book). They would
    otherwise fake TP/SL touches. The cap is the bar's own open/close."""
    o, c = df["open"], df["close"]
    pc, nc = c.shift(1).fillna(o), c.shift(-1).fillna(c)
    hi_ref = np.maximum(pc, nc)
    lo_ref = np.minimum(pc, nc)
    up = df["high"] > hi_ref * (1 + limit)
    dn = df["low"] < lo_ref * (1 - limit)
    df = df.copy()
    df.loc[up, "high"] = np.maximum(o, c)[up]
    df.loc[dn, "low"] = np.minimum(o, c)[dn]
    df.attrs["clipped_wicks"] = int(up.sum() + dn.sum())
    return df


def gap_report(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Rows where the next bar is more than one interval away."""
    step = pd.Timedelta(freq)
    delta = df.index.to_series().diff()
    gaps = delta[delta > step]
    return pd.DataFrame({"gap_end": gaps.index, "missing_bars": (gaps / step - 1).astype(int).values})


def sign_randomized(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Null-hypothesis market: each 1m bar is kept or mirrored (up <-> down) at
    random. Bar sizes, intrabar ranges, volume and volatility clustering stay
    exactly as in the real data, but the direction of every minute is a coin
    flip, so no indicator can have a real edge on it."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    flip = rng.random(len(df)) < 0.5
    rc = np.where(flip, o / c, c / o)
    rh = np.where(flip, o / l, h / o)
    rl = np.where(flip, o / h, l / o)
    new_c = o[0] * np.exp(np.cumsum(np.log(rc)))
    new_o = np.concatenate([[o[0]], new_c[:-1]])
    out = df.copy()
    out["open"], out["close"], out["high"], out["low"] = new_o, new_c, new_o * rh, new_o * rl
    v, tb = df["volume"].to_numpy(), df["taker_buy_volume"].to_numpy()
    out["taker_buy_volume"] = np.where(flip, v - tb, tb)
    return out


def resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 1m bars to `rule`. Index becomes the bar *close* time, i.e. the
    moment the bar is complete and its values can be used without lookahead."""
    agg = df1m.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum", "count": "sum", "taker_buy_volume": "sum",
    }).dropna(subset=["open"])
    agg.index = agg.index + pd.Timedelta(rule)
    agg.index.name = "close_time"
    return agg
