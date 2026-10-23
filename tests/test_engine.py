import numpy as np
import pandas as pd

from btcbot.engine import LOSS, TIMEOUT, WIN, compute_outcomes, simulate


def bars(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def test_long_and_short_resolution():
    # entry at open of bar 0 = 100; long TP 101, SL 99.75; short TP 99, SL 100.25
    o, h, l, c = bars([
        [100.0, 100.2, 99.9, 100.1],
        [100.1, 100.6, 100.0, 100.5],
        [100.5, 101.0, 100.4, 100.9],   # long TP touched here
        [100.9, 100.9, 100.0, 100.1],
    ])
    out = compute_outcomes(o, h, l, c, tp=0.01, sl=0.0025, max_hold=10)
    assert out.result[0, 0] == WIN and out.exit_idx[0, 0] == 2
    assert out.result[1, 0] == LOSS and out.exit_idx[1, 0] == 1   # 100.6 >= 100.25
    assert out.ret[0, 0] == np.float32(0.01) and out.ret[1, 0] == np.float32(-0.0025)


def test_same_bar_tp_and_sl_counts_as_loss():
    o, h, l, c = bars([
        [100.0, 101.5, 99.0, 100.0],
        [100.0, 100.0, 100.0, 100.0],
    ])
    out = compute_outcomes(o, h, l, c, tp=0.01, sl=0.0025, max_hold=10)
    assert out.result[0, 0] == LOSS and out.ambiguous[0, 0]
    assert out.result[1, 0] == LOSS and out.ambiguous[1, 0]


def test_timeout_and_unresolved():
    flat = [[100.0, 100.1, 99.9, 100.05]] * 6
    o, h, l, c = bars(flat)
    out = compute_outcomes(o, h, l, c, tp=0.01, sl=0.0025, max_hold=3)
    assert out.result[0, 0] == TIMEOUT and out.exit_idx[0, 0] == 2
    assert np.isclose(out.ret[0, 0], 0.0005, atol=1e-6)
    assert out.exit_idx[0, 5] == -1          # not enough bars left to decide


def test_one_position_at_a_time():
    o, h, l, c = bars([[100.0, 100.0, 100.0, 100.0]] * 10)
    h[3] = 101.5                                  # every long entered at 0..3 wins at bar 3
    out = compute_outcomes(o, h, l, c, tp=0.01, sl=0.0025, max_hold=20)
    entry = np.arange(10)
    sig = np.zeros(10, bool)
    sig[[0, 1, 3, 5]] = True
    tr = simulate(entry, sig, np.zeros(10, bool), out)
    # 0 exits at 3, so 1 and 3 are skipped; 5 never resolves before the data ends
    assert list(tr.entry_min) == [0]
    assert tr.result[0] == WIN


def test_conflicting_signals_are_skipped():
    o, h, l, c = bars([[100.0, 100.0, 100.0, 100.0]] * 5)
    out = compute_outcomes(o, h, l, c, max_hold=2)
    both = np.array([True, False, False, False, False])
    tr = simulate(np.arange(5), both, both, out)
    assert len(tr) == 0


def test_random_walk_win_rate_matches_theory():
    """Driftless walk: P(+1% before -0.25%) = 0.25 / 1.25 = 20%."""
    rng = np.random.default_rng(0)
    ticks = 20
    steps = rng.normal(0, 0.0004 / np.sqrt(ticks), 200_000 * ticks)
    px = 100 * np.exp(np.cumsum(steps)).reshape(-1, ticks)
    o, c = px[:, 0], px[:, -1]
    h, l = px.max(axis=1), px.min(axis=1)
    out = compute_outcomes(o, h, l, c, max_hold=100_000)
    sample = out.exit_idx[0, ::50] >= 0
    wr = (out.result[0, ::50][sample] == WIN).mean()
    assert 0.17 < wr < 0.23, wr


def test_sign_randomized_keeps_bar_geometry():
    from btcbot.data import sign_randomized
    from tests.test_conditions import synthetic_1m
    df = synthetic_1m(n=5000)
    null = sign_randomized(df, seed=1)
    assert (null["high"] >= null[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (null["low"] <= null[["open", "close"]].min(axis=1) + 1e-9).all()
    real_abs = np.abs(np.log(df["close"] / df["open"]))
    null_abs = np.abs(np.log(null["close"] / null["open"]))
    assert np.allclose(real_abs, null_abs)
    assert np.allclose(np.log(df["high"] / df["low"]), np.log(null["high"] / null["low"]))
