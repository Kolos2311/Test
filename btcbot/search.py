"""Brute-force screening of condition combinations with packed bitsets.

Screening counts *signal bars* (overlapping): for each combination it returns
how many bars fire long/short and how many of those would have hit TP first.
It is only a fast filter; candidates are then re-evaluated with the exact
one-position-at-a-time simulation in `engine.simulate`.
"""
from __future__ import annotations

import itertools

import numpy as np
from numba import njit, prange, uint64

from .conditions import Condition

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


@njit(inline="always")
def _popcount(x):
    x = x - ((x >> uint64(1)) & _M1)
    x = (x & _M2) + ((x >> uint64(2)) & _M2)
    x = (x + (x >> uint64(4))) & _M4
    return (x * _H01) >> uint64(56)


def pack(mask: np.ndarray) -> np.ndarray:
    """bool[n] -> uint64[ceil(n/64)]"""
    pad = (-len(mask)) % 64
    m = np.concatenate([mask.astype(np.uint8), np.zeros(pad, np.uint8)])
    return np.packbits(m, bitorder="little").view(np.uint64)


@njit(parallel=True, cache=True)
def _screen(combos, L, S, WL, WS):
    n_combo, width = combos.shape
    n_words = L.shape[1]
    out = np.zeros((n_combo, 4), dtype=np.int64)
    for ci in prange(n_combo):
        nl = 0
        wl = 0
        ns = 0
        ws = 0
        for w in range(n_words):
            ml = L[combos[ci, 0], w]
            ms = S[combos[ci, 0], w]
            for j in range(1, width):
                c = combos[ci, j]
                if c >= 0:
                    ml &= L[c, w]
                    ms &= S[c, w]
            nl += np.int64(_popcount(ml))
            wl += np.int64(_popcount(ml & WL[w]))
            ns += np.int64(_popcount(ms))
            ws += np.int64(_popcount(ms & WS[w]))
        out[ci, 0] = nl
        out[ci, 1] = wl
        out[ci, 2] = ns
        out[ci, 3] = ws
    return out


def screen(combos: np.ndarray, conds: list[Condition], sample: np.ndarray,
           win_long: np.ndarray, win_short: np.ndarray) -> np.ndarray:
    """Signal-bar counts [n_long, wins_long, n_short, wins_short] per combination,
    restricted to bars where `sample` is True."""
    L = np.stack([pack(c.long & sample) for c in conds])
    S = np.stack([pack(c.short & sample) for c in conds])
    return _screen(combos.astype(np.int64), L, S, pack(win_long & sample), pack(win_short & sample))


def enumerate_combos(conds: list[Condition], max_size: int = 3) -> np.ndarray:
    """All combinations of 1..max_size conditions from distinct families that
    contain at least one directional condition. Rows padded with -1."""
    fam = [c.family for c in conds]
    rows = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(len(conds)), size):
            if len({fam[i] for i in combo}) < size:
                continue
            if not any(conds[i].directional for i in combo):
                continue
            rows.append(list(combo) + [-1] * (max_size - size))
    return np.array(rows, dtype=np.int64)


def extend_combos(base: np.ndarray, conds: list[Condition], width: int) -> np.ndarray:
    """Add one more condition (distinct family) to each base combination."""
    fam = [c.family for c in conds]
    seen = set()
    rows = []
    for row in base:
        idx = [int(i) for i in row if i >= 0]
        used = {fam[i] for i in idx}
        for j in range(len(conds)):
            if fam[j] in used:
                continue
            key = tuple(sorted(idx + [j]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(list(key) + [-1] * (width - len(key)))
    return np.array(rows, dtype=np.int64).reshape(-1, width)


def combo_mask(row, conds: list[Condition]):
    idx = [int(i) for i in row if i >= 0]
    lg = np.logical_and.reduce([conds[i].long for i in idx])
    sh = np.logical_and.reduce([conds[i].short for i in idx])
    return lg, sh


def combo_name(row, conds: list[Condition]) -> str:
    return " + ".join(conds[int(i)].name for i in row if i >= 0)
