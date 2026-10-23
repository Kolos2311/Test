"""Trade statistics. Returns are in percent of entry price, no leverage."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .engine import LOSS, TIMEOUT, WIN, Trades

# Hyperliquid perps, base tier: taker 0.045%, maker 0.015% per side.
FEE_TAKER_RT = 0.09   # % round trip, market entry + market/stop exit
FEE_MAKER_RT = 0.03   # % round trip, limit entry + limit TP (SL would still be taker)


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def binom_p(wins: int, n: int, p0: float) -> float:
    """One-sided p-value of observing >= wins under win probability p0."""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(wins - 1, n, p0))


def summarize(tr: Trades, minute_index: pd.DatetimeIndex, fee_rt: float = FEE_TAKER_RT) -> dict:
    n = len(tr)
    if n == 0:
        return {"trades": 0}
    wins = int((tr.result == WIN).sum())
    losses = int((tr.result == LOSS).sum())
    timeouts = int((tr.result == TIMEOUT).sum())
    gross = tr.ret * 100
    net = gross - fee_rt
    eq = np.cumsum(net)
    dd = float((np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq).max())
    t0, t1 = minute_index[tr.entry_min[0]], minute_index[tr.entry_min[-1]]
    months = max((t1 - t0).days / 30.44, 1.0)
    lo, hi = wilson(wins, n)
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    longs = tr.direction == 0
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / n,
        "wr_ci_low": lo,
        "wr_ci_high": hi,
        "long_trades": int(longs.sum()),
        "long_wr": float((tr.result[longs] == WIN).mean()) if longs.any() else np.nan,
        "short_trades": int((~longs).sum()),
        "short_wr": float((tr.result[~longs] == WIN).mean()) if (~longs).any() else np.nan,
        "exp_gross_pct": float(gross.mean()),
        "exp_net_pct": float(net.mean()),
        "sum_gross_pct": float(gross.sum()),
        "sum_net_pct": float(net.sum()),
        "profit_factor_net": float(pos / neg) if neg > 0 else np.inf,
        "max_dd_net_pct": dd,
        "trades_per_month": n / months,
        "avg_hold_min": float(np.mean(tr.exit_min - tr.entry_min + 1)),
        "ambiguous": int(tr.ambiguous.sum()),
    }


def yearly(tr: Trades, minute_index: pd.DatetimeIndex) -> pd.DataFrame:
    if len(tr) == 0:
        return pd.DataFrame()
    years = minute_index[tr.entry_min].year
    df = pd.DataFrame({"year": years, "win": tr.result == WIN, "ret": tr.ret * 100})
    g = df.groupby("year")
    return pd.DataFrame({"trades": g.size(), "win_rate": g["win"].mean(),
                         "sum_gross_pct": g["ret"].sum()})
