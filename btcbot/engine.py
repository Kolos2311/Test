"""Fixed take-profit / stop-loss trade resolution on 1m bars.

Trade model (both directions, no leverage, returns in % of entry price):
  * signal is evaluated on the close of a timeframe bar;
  * entry at the open of the next 1m bar;
  * long:  TP = entry * (1 + tp), SL = entry * (1 - sl); short mirrored;
  * walking forward minute by minute, the first level touched decides the trade;
  * if TP and SL are both inside the same 1m bar the order is unknown and the
    trade is counted as a LOSS (conservative);
  * a trade still open after `max_hold` minutes is closed at that bar's close;
  * one position at a time: signals while a trade is open are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

WIN, LOSS, TIMEOUT = 1, -1, 0


@njit(cache=True)
def _resolve(o, h, l, c, tp, sl, max_hold, direction):
    n = len(o)
    result = np.zeros(n, dtype=np.int8)
    exit_idx = np.full(n, -1, dtype=np.int64)
    ret = np.zeros(n, dtype=np.float32)
    ambiguous = np.zeros(n, dtype=np.bool_)
    for j in range(n):
        e = o[j]
        if direction == 1:
            tp_px, sl_px = e * (1 + tp), e * (1 - sl)
        else:
            tp_px, sl_px = e * (1 - tp), e * (1 + sl)
        last = min(n - 1, j + max_hold - 1)
        done = False
        for k in range(j, last + 1):
            if direction == 1:
                hit_tp, hit_sl = h[k] >= tp_px, l[k] <= sl_px
            else:
                hit_tp, hit_sl = l[k] <= tp_px, h[k] >= sl_px
            if hit_sl:
                result[j], exit_idx[j], ret[j] = LOSS, k, -sl
                ambiguous[j] = hit_tp
                done = True
                break
            if hit_tp:
                result[j], exit_idx[j], ret[j] = WIN, k, tp
                done = True
                break
        if not done:
            if last - j + 1 < max_hold:
                # ran out of data: leave unresolved (exit_idx == -1)
                continue
            result[j], exit_idx[j] = TIMEOUT, last
            ret[j] = direction * (c[last] / e - 1)
    return result, exit_idx, ret, ambiguous


@dataclass
class Outcomes:
    """Outcome of entering at the open of every 1m bar, per direction."""
    result: np.ndarray      # int8 [2, n]: row 0 long, row 1 short
    exit_idx: np.ndarray    # int64 [2, n]
    ret: np.ndarray         # float32 [2, n], fraction of entry price
    ambiguous: np.ndarray   # bool [2, n]

    def save(self, path):
        np.savez_compressed(path, result=self.result, exit_idx=self.exit_idx,
                            ret=self.ret, ambiguous=self.ambiguous)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["result"], z["exit_idx"], z["ret"], z["ambiguous"])


def compute_outcomes(o, h, l, c, tp=0.01, sl=0.0025, max_hold=7 * 24 * 60) -> Outcomes:
    parts = [_resolve(o, h, l, c, tp, sl, max_hold, d) for d in (1, -1)]
    return Outcomes(*(np.stack([p[i] for p in parts]) for i in range(4)))


@njit(cache=True)
def _simulate(entry_min, long_sig, short_sig, exit_idx, valid_from, valid_to):
    """Walk the signal bars in time order taking at most one trade at a time.
    Returns indices into the signal arrays and the direction taken."""
    n = len(entry_min)
    taken = np.empty(n, dtype=np.int64)
    dirs = np.empty(n, dtype=np.int8)
    k = 0
    busy_until = -1
    for i in range(n):
        if not (long_sig[i] or short_sig[i]) or (long_sig[i] and short_sig[i]):
            continue
        m = entry_min[i]
        if m < 0 or m <= busy_until or m < valid_from or m >= valid_to:
            continue
        d = 0 if long_sig[i] else 1
        x = exit_idx[d, m]
        if x < 0:
            continue
        taken[k] = i
        dirs[k] = d
        k += 1
        busy_until = x
    return taken[:k], dirs[:k]


@dataclass
class Trades:
    entry_min: np.ndarray   # 1m index of entry
    exit_min: np.ndarray
    direction: np.ndarray   # 0 long, 1 short
    result: np.ndarray      # WIN / LOSS / TIMEOUT
    ret: np.ndarray         # gross return fraction
    ambiguous: np.ndarray

    def __len__(self):
        return len(self.entry_min)

    def subset(self, mask):
        return Trades(*(getattr(self, f)[mask] for f in
                        ("entry_min", "exit_min", "direction", "result", "ret", "ambiguous")))


def simulate(entry_min, long_sig, short_sig, out: Outcomes,
             valid_from: int = 0, valid_to: int | None = None) -> Trades:
    valid_to = out.exit_idx.shape[1] if valid_to is None else valid_to
    idx, dirs = _simulate(entry_min, long_sig.astype(np.bool_), short_sig.astype(np.bool_),
                          out.exit_idx, valid_from, valid_to)
    m = entry_min[idx]
    return Trades(m, out.exit_idx[dirs, m], dirs, out.result[dirs, m],
                  out.ret[dirs, m].astype(np.float64), out.ambiguous[dirs, m])
