"""Walk-forward test of the selection procedure itself.

Rolling windows: select the top 5 combinations on 24 months, trade them on the
next 6 months, roll forward by 6 months. The pooled out-of-sample trades show
whether mining indicator combinations produces an edge on unseen data at all.
Run it on real data and on sign-randomized (null) data and compare.

Two ways to rank candidates inside a training window:
  lb          - Wilson lower bound of the pooled train win rate;
  consistency - worst win rate over the half-years of the window
                (a combination must work in every half-year, not on average).

    python -m research.walk_forward --data BTCUSD-bitstamp [--null-seed 1]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import data, metrics
from btcbot.engine import WIN
from btcbot.search import combo_name

from . import run_research as rr

TRAIN_MONTHS, TEST_MONTHS = 24, 6
FIRST = "2020-01-01"
MIN_HALF_TRADES = 8


def windows(end: pd.Timestamp, train_months: int = TRAIN_MONTHS):
    out = []
    t0 = pd.Timestamp(FIRST, tz="UTC")
    while True:
        tr_end = t0 + pd.DateOffset(months=train_months)
        if tr_end >= end:
            return out
        out.append((t0, tr_end, min(tr_end + pd.DateOffset(months=TEST_MONTHS), end)))
        t0 += pd.DateOffset(months=TEST_MONTHS)


def pick_top(df: pd.DataFrame, score: str, k: int = 5) -> pd.DataFrame:
    top = []
    for _, r in df.sort_values(score, ascending=False).iterrows():
        fam = set(r["families"])
        if any(t["tf"] == r["tf"] and len(fam & set(t["families"])) / len(fam | set(t["families"])) >= 0.5
               for t in top):
            continue
        top.append(r)
        if len(top) == k:
            break
    return pd.DataFrame(top)


def candidates(frames: dict, out, idx: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    """Screen and simulate every timeframe on the training span [a, b); returns
    candidates passing the trade-count filters with both ranking scores."""
    va, vb = int(idx.searchsorted(a)), int(idx.searchsorted(b))
    n_halves = max(1, round((b - a).days / 182.6))
    halves = [int(idx.searchsorted(a + (b - a) * i / n_halves)) for i in range(n_halves + 1)]
    rows = []
    for tf, fr in frames.items():
        train = np.asarray((fr.index >= a) & (fr.index < b))
        cand, _ = rr.screen_frame(fr, train)
        for row in cand:
            tr = rr.run_combo_between(fr, row, out, va, vb)
            n = len(tr)
            if n < rr.MIN_TRAIN_TRADES:
                continue
            longs = int((tr.direction == 0).sum())
            if longs < rr.MIN_SIDE_TRADES or n - longs < rr.MIN_SIDE_TRADES:
                continue
            wins = int((tr.result == WIN).sum())
            half_wr = []
            for h0, h1 in zip(halves[:-1], halves[1:]):
                m = (tr.entry_min >= h0) & (tr.entry_min < h1)
                half_wr.append(float((tr.result[m] == WIN).mean()) if m.sum() >= MIN_HALF_TRADES else 0.0)
            rows.append({"tf": tf, "combo": tuple(int(i) for i in row if i >= 0),
                         "name": combo_name(row, fr.conds),
                         "conditions": [fr.conds[int(i)].name for i in row if i >= 0],
                         "families": sorted({fr.conds[int(i)].family for i in row if i >= 0}),
                         "train_trades": n, "train_wr": wins / n,
                         "train_long_trades": longs, "train_half_wr": half_wr,
                         "train_exp_gross_pct": float(tr.ret.mean() * 100),
                         "lb": metrics.wilson(wins, n)[0], "consistency": min(half_wr)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="BTCUSDT")
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--timeframes", default=",".join(rr.TIMEFRAMES))
    ap.add_argument("--out", default="reports/walk_forward")
    ap.add_argument("--train-months", type=int, default=TRAIN_MONTHS)
    ap.add_argument("--min-trades", type=int, default=rr.MIN_TRAIN_TRADES)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rr.TOP_BY_LB, rr.TOP_BY_WR, rr.BEAM_KEEP = 800, 200, 300
    rr.MIN_TRAIN_TRADES = args.min_trades
    train_months = args.train_months

    df1m = data.load(args.data)
    tag = args.data
    if args.null_seed is not None:
        df1m = data.sign_randomized(df1m, args.null_seed)
        tag = f"{args.data}-null{args.null_seed}"
    idx = df1m.index
    out = rr.load_outcomes(df1m, tag)
    frames = {tf: rr.prepare(tf, rr.TIMEFRAMES[tf], df1m, out) for tf in args.timeframes.split(",")}
    print(f"[{tag}] prepared {len(frames)} timeframes")

    picks, oos_trades = [], []
    for w, (a, b, c) in enumerate(windows(idx[-1], train_months)):
        t = time.time()
        vb, vc = int(idx.searchsorted(b)), int(idx.searchsorted(c))
        cand_df = candidates(frames, out, idx, a, b)
        for score in ("lb", "consistency"):
            for rank, (_, r) in enumerate(pick_top(cand_df, score).iterrows()):
                tr = rr.run_combo_between(frames[r["tf"]], list(r["combo"]), out, vb, vc)
                picks.append({"window": w, "train_from": a.date(), "test_from": b.date(), "test_to": c.date(),
                              "score": score, "rank": rank + 1, "tf": r["tf"], "name": r["name"],
                              "train_trades": r["train_trades"], "train_wr": r["train_wr"],
                              "train_consistency": r["consistency"],
                              "test_trades": len(tr), "test_wins": int((tr.result == WIN).sum())})
                for d, res, ret in zip(tr.direction, tr.result, tr.ret):
                    oos_trades.append({"window": w, "score": score, "rank": rank + 1,
                                       "dir": int(d), "win": bool(res == WIN), "ret_pct": float(ret) * 100})
        print(f"  window {w}: train {a.date()}..{b.date()} test ..{c.date()}  "
              f"{len(cand_df)} candidates  {time.time() - t:.0f}s")

    picks = pd.DataFrame(picks)
    trades = pd.DataFrame(oos_trades)
    picks.to_csv(out_dir / f"picks_{tag}.csv", index=False)
    trades.to_csv(out_dir / f"oos_trades_{tag}.csv", index=False)

    # baseline = enter every minute in the out-of-sample spans
    ws = windows(idx[-1], train_months)
    oos_from, oos_to = int(idx.searchsorted(ws[0][1])), len(idx)
    res = out.result[:, oos_from:oos_to]
    ok = out.exit_idx[:, oos_from:oos_to] >= 0
    base = float(np.concatenate([res[0][ok[0]], res[1][ok[1]]]).__eq__(WIN).mean())
    summary = {"dataset": tag, "train_months": train_months, "min_trades": rr.MIN_TRAIN_TRADES,
               "oos_from": str(ws[0][1].date()), "baseline_wr": base}
    for score, g in trades.groupby("score"):
        n, wins = len(g), int(g["win"].sum())
        net = g["ret_pct"] - metrics.FEE_TAKER_RT
        lo, hi = metrics.wilson(wins, n)
        summary[score] = {"trades": n, "win_rate": wins / n, "wr_ci95": [lo, hi],
                          "p_value_vs_baseline": metrics.binom_p(wins, n, base),
                          "exp_gross_pct": float(g["ret_pct"].mean()), "exp_net_pct": float(net.mean()),
                          "sum_net_pct": float(net.sum())}
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
