import numpy as np
import pandas as pd

from btcbot import conditions, data
from btcbot.search import combo_mask, enumerate_combos, screen


def synthetic_1m(n=60 * 24 * 40, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0005, n)) * close
    vol = rng.lognormal(3, 0.5, n)
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": vol, "quote_volume": vol * close, "count": 100.0,
        "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n),
    }, index=idx)


def test_resample_labels_bar_close():
    df = synthetic_1m(n=180)
    h = data.resample(df, "1h")
    assert h.index[0] == pd.Timestamp("2021-01-01 01:00", tz="UTC")
    assert h["open"].iloc[0] == df["open"].iloc[0]
    assert h["close"].iloc[0] == df["close"].iloc[59]
    assert h["high"].iloc[0] == df["high"].iloc[:60].max()


def test_conditions_have_no_lookahead():
    """Every condition at bar t must be identical whether or not later data exist."""
    df = synthetic_1m()
    full_tf = data.resample(df, "15min")
    cut = df.index[len(df) * 3 // 4]
    part = df[df.index < cut]
    part_tf = data.resample(part, "15min")
    # the last 15m bar of `part` may be complete in both; compare bars that are complete in part
    n = len(part_tf) - 1

    full = conditions.build(full_tf, {lbl: data.resample(df, lbl) for lbl in ("1h", "4h")})
    trunc = conditions.build(part_tf, {lbl: data.resample(part, lbl) for lbl in ("1h", "4h")})
    assert [c.name for c in full] == [c.name for c in trunc]
    for a, b in zip(full, trunc):
        assert np.array_equal(a.long[:n], b.long[:n]), a.name
        assert np.array_equal(a.short[:n], b.short[:n]), a.name


def test_screen_matches_direct_count():
    df = synthetic_1m(n=60 * 24 * 10)
    tf = data.resample(df, "5min")
    conds = conditions.build(tf)
    rng = np.random.default_rng(3)
    sample = rng.random(len(tf)) < 0.9
    wl, ws = rng.random(len(tf)) < 0.2, rng.random(len(tf)) < 0.2
    combos = enumerate_combos(conds, 2)[::97]
    got = screen(combos, conds, sample, wl, ws)
    for row, g in zip(combos, got):
        lg, sh = combo_mask(row, conds)
        lg, sh = lg & sample, sh & sample
        assert list(g) == [lg.sum(), (lg & wl).sum(), sh.sum(), (sh & ws).sum()]
