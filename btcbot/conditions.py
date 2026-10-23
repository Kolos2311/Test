"""Library of boolean entry conditions built from indicators.

Each directional condition has a long rule and its mirrored short rule
(e.g. RSI < 30 for long, RSI > 70 for short). Non-directional conditions
(volume spike, volatility regime, session, ...) apply the same mask to both
sides and only make sense combined with at least one directional condition.
`family` groups variants of the same idea so a combination never stacks two
thresholds of one indicator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Condition:
    name: str
    family: str
    long: np.ndarray
    short: np.ndarray
    directional: bool


def _b(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.fillna(False).to_numpy(dtype=bool)
    return np.asarray(x, dtype=bool)


def _cross_up(a, b):
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a > b) & (a.shift(1) <= b_prev)


def _cross_down(a, b):
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return (a < b) & (a.shift(1) >= b_prev)


class _Builder:
    def __init__(self):
        self.items: list[Condition] = []

    def d(self, name, family, long, short):
        self.items.append(Condition(name, family, _b(long), _b(short), True))

    def n(self, name, family, mask):
        m = _b(mask)
        self.items.append(Condition(name, family, m, m, False))


def build(df: pd.DataFrame, htf: dict[str, pd.DataFrame] | None = None) -> list[Condition]:
    """Conditions evaluated on the close of each bar of `df` (indexed by close time).
    `htf` maps a label (e.g. '4h') to a higher-timeframe frame; its values are
    joined as of the last *completed* higher-timeframe bar."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    B = _Builder()

    e9, e20, e21, e50, e200 = (ind.ema(c, n) for n in (9, 20, 21, 50, 200))
    m_line, m_sig = ind.macd(c)
    st = ind.supertrend(h, l, c, 10, 3.0)
    ps = ind.psar_trend(h, l)
    cloud_top, cloud_bot = ind.ichimoku_cloud(h, l)
    pdi, mdi, adx = ind.dmi_adx(h, l, c, 14)
    vw = ind.session_vwap(df)
    a14 = ind.atr(h, l, c, 14)
    r14, r2 = ind.rsi(c, 14), ind.rsi(c, 2)
    bb_mid, bb_up, bb_lo = ind.bollinger(c, 20, 2.0)
    kc_mid, kc_up, kc_lo = ind.keltner(h, l, c, 20, 2.0)
    k, dd = ind.stochastic(h, l, c)
    wr = ind.williams_r(h, l, c)
    cc = ind.cci(h, l, c)
    mf = ind.mfi(h, l, c, v)
    z50 = ind.zscore(c, 50)
    sk = ind.streak(c)
    ob = ind.obv(c, v)
    ob_e = ind.ema(ob, 20)
    tbr = df["taker_buy_volume"] / v.replace(0, np.nan)
    tbr3 = df["taker_buy_volume"].rolling(3).sum() / v.rolling(3).sum().replace(0, np.nan)
    ret3 = c / c.shift(3) - 1
    sd3 = ret3.rolling(500, min_periods=200).std()
    body = (c - o).abs()
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)

    # --- trend state ---
    B.d("close>EMA50", "trend_ema50", c > e50, c < e50)
    B.d("close>EMA200", "trend_ema200", c > e200, c < e200)
    B.d("EMA20>EMA50>EMA200", "ema_stack", (e20 > e50) & (e50 > e200), (e20 < e50) & (e50 < e200))
    B.d("EMA9>EMA21", "ema_fast", e9 > e21, e9 < e21)
    B.d("EMA50 rising(10)", "ema_slope", e50 > e50.shift(10), e50 < e50.shift(10))
    B.d("MACD>signal", "macd_sig", m_line > m_sig, m_line < m_sig)
    B.d("MACD>0", "macd_zero", m_line > 0, m_line < 0)
    B.d("Supertrend(10,3) up", "supertrend", st == 1, st == -1)
    B.d("PSAR up", "psar", ps == 1, ps == -1)
    B.d("close above Ichimoku cloud", "ichimoku", c > cloud_top, c < cloud_bot)
    B.d("+DI>-DI", "dmi", pdi > mdi, pdi < mdi)
    B.d("close>day VWAP", "vwap_side", c > vw, c < vw)

    # --- momentum / breakout events ---
    B.d("EMA9 x EMA21", "ema_fast_x", _cross_up(e9, e21), _cross_down(e9, e21))
    B.d("MACD x signal", "macd_x", _cross_up(m_line, m_sig), _cross_down(m_line, m_sig))
    B.d("Supertrend flip", "supertrend_flip", (st == 1) & (st.shift(1) == -1), (st == -1) & (st.shift(1) == 1))
    B.d("Donchian20 breakout", "donchian", c > h.rolling(20).max().shift(1), c < l.rolling(20).min().shift(1))
    B.d("Donchian55 breakout", "donchian", c > h.rolling(55).max().shift(1), c < l.rolling(55).min().shift(1))
    B.d("close>BB upper (breakout)", "bb_break", c > bb_up, c < bb_lo)
    B.d("RSI14>60 (momentum)", "rsi_mom", r14 > 60, r14 < 40)
    B.d("RSI14>70 (momentum)", "rsi_mom", r14 > 70, r14 < 30)
    B.d("3-bar burst >1.5sd", "mom_burst", ret3 > 1.5 * sd3, ret3 < -1.5 * sd3)
    B.d("candle body >1 ATR", "big_candle", (c - o) > a14, (o - c) > a14)
    B.d("bull candle", "candle_dir", c > o, c < o)
    B.d("engulfing", "engulf",
        (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1)),
        (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1)))

    # --- mean reversion ---
    B.d("RSI14<30", "rsi14_os", r14 < 30, r14 > 70)
    B.d("RSI14<25", "rsi14_os", r14 < 25, r14 > 75)
    B.d("RSI14<20", "rsi14_os", r14 < 20, r14 > 80)
    B.d("RSI2<10", "rsi2", r2 < 10, r2 > 90)
    B.d("RSI2<5", "rsi2", r2 < 5, r2 > 95)
    B.d("RSI14 x up 30", "rsi_x", _cross_up(r14, 30), _cross_down(r14, 70))
    B.d("Stoch K x D in OS", "stoch_x",
        _cross_up(k, dd) & (k.shift(1) < 20), _cross_down(k, dd) & (k.shift(1) > 80))
    B.d("Stoch K<10", "stoch_lvl", k < 10, k > 90)
    B.d("Williams%R<-90", "williams", wr < -90, wr > -10)
    B.d("CCI<-100", "cci", cc < -100, cc > 100)
    B.d("CCI<-200", "cci", cc < -200, cc > 200)
    B.d("MFI<20", "mfi", mf < 20, mf > 80)
    B.d("close<BB lower", "bb_lo", c < bb_lo, c > bb_up)
    B.d("BB re-entry", "bb_reentry",
        (c.shift(1) < bb_lo.shift(1)) & (c > bb_lo), (c.shift(1) > bb_up.shift(1)) & (c < bb_up))
    B.d("close<Keltner lower", "keltner", c < kc_lo, c > kc_up)
    B.d("z50<-2", "zscore", z50 < -2, z50 > 2)
    B.d("z50<-3", "zscore", z50 < -3, z50 > 3)
    B.d("close<EMA20", "ema20_dev", c < e20, c > e20)
    B.d("close<VWAP-1ATR", "vwap_dev", c < vw - a14, c > vw + a14)
    B.d(">=4 down closes", "streak", sk <= -4, sk >= 4)
    B.d("long lower wick", "wick",
        (lower_wick > 2 * body) & (lower_wick > 0.5 * a14),
        (upper_wick > 2 * body) & (upper_wick > 0.5 * a14))

    # --- order flow / volume ---
    B.d("taker buy >55%", "taker", tbr > 0.55, tbr < 0.45)
    B.d("taker buy >60%", "taker", tbr > 0.60, tbr < 0.40)
    B.d("taker buy 3-bar >55%", "taker3", tbr3 > 0.55, tbr3 < 0.45)
    B.d("taker buy <42% (exhaustion)", "taker_contra", tbr < 0.42, tbr > 0.58)
    B.d("OBV>EMA20(OBV)", "obv", ob > ob_e, ob < ob_e)

    # --- non-directional regime / context ---
    vs = v / ind.sma(v, 20)
    atr_rank = ind.rolling_pct_rank(a14 / c, 500)
    bbw = (bb_up - bb_lo) / bb_mid
    bbw_rank = ind.rolling_pct_rank(bbw, 500)
    B.n("volume >2x avg", "volspike", vs > 2)
    B.n("volume >3x avg", "volspike", vs > 3)
    B.n("ADX>25", "adx", adx > 25)
    B.n("ADX<20", "adx", adx < 20)
    B.n("ATR% top 20%", "atr_regime", atr_rank > 0.8)
    B.n("ATR% bottom 20%", "atr_regime", atr_rank < 0.2)
    B.n("BB squeeze (width bottom 10%)", "squeeze", bbw_rank < 0.1)
    B.n("squeeze release", "squeeze", (bbw_rank.shift(1).rolling(10).min() < 0.1) & (bbw_rank > 0.3))
    hour = pd.Series(df.index.hour, index=df.index)
    B.n("session Asia 00-08 UTC", "session", hour < 8)
    B.n("session EU 08-13 UTC", "session", (hour >= 8) & (hour < 13))
    B.n("session US 13-21 UTC", "session", (hour >= 13) & (hour < 21))
    B.n("session late 21-24 UTC", "session", hour >= 21)
    dow = pd.Series(df.index.dayofweek, index=df.index)
    B.n("weekend", "weekday", dow >= 5)
    B.n("weekday", "weekday", dow < 5)

    # --- higher timeframe context ---
    for label, hdf in (htf or {}).items():
        hc = hdf["close"]
        h_e50 = ind.ema(hc, 50)
        h_st = ind.supertrend(hdf["high"], hdf["low"], hc, 10, 3.0)
        h_r = ind.rsi(hc, 14)
        al = lambda s: s.reindex(df.index, method="ffill")  # noqa: E731
        B.d(f"{label} close>EMA50", f"htf_{label}_trend", al(hc > h_e50), al(hc < h_e50))
        B.d(f"{label} Supertrend up", f"htf_{label}_st", al(h_st == 1), al(h_st == -1))
        B.d(f"{label} RSI>50", f"htf_{label}_rsi", al(h_r > 50), al(h_r < 50))
        B.d(f"{label} RSI<30", f"htf_{label}_rsi_os", al(h_r < 30), al(h_r > 70))

    return B.items
