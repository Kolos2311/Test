"""Technical indicators on pandas Series. Every value at index t uses only
bars up to and including t (no lookahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    out = x.ewm(span=n, adjust=False, min_periods=n).mean()
    return out


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def keltner(high, low, close, n: int = 20, k: float = 2.0):
    mid = ema(close, n)
    a = atr(high, low, close, n)
    return mid, mid + k * a, mid - k * a


def stochastic(high, low, close, n: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = sma(raw, smooth_k)
    d = sma(k, smooth_d)
    return k, d


def williams_r(high, low, close, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


@njit(cache=True)
def _mean_abs_dev(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        out[i] = np.mean(np.abs(w - w.mean()))
    return out


def cci(high, low, close, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    m = sma(tp, n)
    md = pd.Series(_mean_abs_dev(tp.to_numpy(), n), index=tp.index)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def mfi(high, low, close, volume, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    flow = tp * volume
    d = tp.diff()
    pos = flow.where(d > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = flow.where(d < 0, 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def dmi_adx(high, low, close, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(high, low, close, n)
    plus_di = 100 * wilder(plus_dm, n) / a
    minus_di = 100 * wilder(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return plus_di, minus_di, wilder(dx, n)


@njit(cache=True)
def _supertrend(h, l, c, a, mult):
    n = len(c)
    direction = np.zeros(n, dtype=np.int8)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        mid = (h[i] + l[i]) / 2
        bu = mid + mult * a[i]
        bl = mid - mult * a[i]
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i], lower[i] = bu, bl
            direction[i] = 1 if c[i] >= mid else -1
            continue
        upper[i] = bu if (bu < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = bl if (bl > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < lower[i] else 1
        else:
            direction[i] = 1 if c[i] > upper[i] else -1
    return direction


def supertrend(high, low, close, n: int = 10, mult: float = 3.0) -> pd.Series:
    a = atr(high, low, close, n).to_numpy()
    d = _supertrend(high.to_numpy(), low.to_numpy(), close.to_numpy(), a, mult)
    return pd.Series(d, index=close.index)


@njit(cache=True)
def _psar(h, l, step, max_step):
    n = len(h)
    trend = np.zeros(n, dtype=np.int8)
    if n < 2:
        return trend
    up = h[1] >= h[0]
    sar = l[0] if up else h[0]
    ep = h[1] if up else l[1]
    af = step
    trend[1] = 1 if up else -1
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        trend[i] = 1 if up else -1
    return trend


def psar_trend(high, low, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    return pd.Series(_psar(high.to_numpy(), low.to_numpy(), step, max_step), index=high.index)


def ichimoku_cloud(high, low, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    """Cloud bounds valid at t: spans are projected `kijun` bars forward, so the
    value shown at t was computed from data ending at t - kijun."""
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((t + k) / 2).shift(kijun)
    span_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    return np.maximum(span_a, span_b), np.minimum(span_a, span_b)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset at 00:00 UTC. Index is bar close time, so a bar closing at
    exactly 00:00 belongs to the previous day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = (df.index - pd.Timedelta("1ns")).floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return pv / v.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std(ddof=0)
    return (x - m) / s.replace(0, np.nan)


def rolling_pct_rank(x: pd.Series, n: int) -> pd.Series:
    """Percentile of the current value within the trailing n-bar window."""
    return x.rolling(n, min_periods=n).rank(pct=True)


def streak(close: pd.Series) -> pd.Series:
    """Signed count of consecutive up (+) or down (-) closes."""
    s = np.sign(close.diff()).fillna(0).to_numpy()
    out = np.zeros(len(s))
    for i in range(1, len(s)):
        if s[i] != 0 and s[i] == np.sign(out[i - 1]):
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return pd.Series(out, index=close.index)
