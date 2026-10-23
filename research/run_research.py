"""Independent search for indicator combinations on BTCUSDT futures with a fixed
take-profit of 1% and stop-loss of 0.25% of price (no leverage), long and short.

Protocol (no peeking at later data when choosing):
  1. train 2020-2023: brute-force screen every combination of 1-3 conditions
     (plus a beam extension to 4) on every timeframe, then re-run the best
     candidates with the exact one-position-at-a-time simulation;
  2. validation 2024: candidates that pass train are re-tested; the final
     ranking uses train+validation trades only;
  3. test 2025-now: the chosen top 5 are run once on unseen data and reported
     as they come out.

    python -m research.run_research [--timeframes 15min,1h] [--out reports]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import conditions, data, metrics
from btcbot.engine import WIN, Outcomes, compute_outcomes, simulate
from btcbot.search import combo_mask, combo_name, enumerate_combos, extend_combos, screen

TP, SL = 0.01, 0.0025
PERIODS = {
    "train": ("2020-01-01", "2024-01-01"),
    "val": ("2024-01-01", "2025-01-01"),
    "test": ("2025-01-01", "2100-01-01"),
}
TIMEFRAMES = {
    "5min": ["1h", "4h", "1D"],
    "15min": ["1h", "4h", "1D"],
    "30min": ["4h", "1D"],
    "1h": ["4h", "1D"],
    "4h": ["1D"],
}
# screening / selection thresholds
MIN_SIG_PER_SIDE = 30      # signal bars per direction in train to be screened in
TOP_BY_LB = 1500           # candidates per timeframe kept by Wilson lower bound
TOP_BY_WR = 300            # ... and by raw win rate (to probe the high-WR tail)
BEAM_BASE, BEAM_KEEP = 300, 500
MIN_TRAIN_TRADES, MIN_SIDE_TRADES = 40, 10
TO_VALIDATION = 400
MIN_VAL_TRADES = 15


@dataclass
class Frame:
    tf: str
    index: pd.DatetimeIndex
    conds: list
    entry_min: np.ndarray
    sample: np.ndarray        # bar has a resolved trade outcome in both directions
    win_long: np.ndarray
    win_short: np.ndarray


def wilson_lb(w, n, z=1.96):
    w, n = np.asarray(w, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = w / n
        lb = (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return np.nan_to_num(lb)


def load_outcomes(df1m: pd.DataFrame, tag: str = "real") -> Outcomes:
    path = data.DATA_DIR / f"outcomes_{tag}_tp{TP}_sl{SL}_{len(df1m)}.npz"
    if path.exists():
        return Outcomes.load(path)
    t = time.time()
    out = compute_outcomes(*(df1m[c].to_numpy() for c in ("open", "high", "low", "close")), tp=TP, sl=SL)
    out.save(path)
    print(f"outcomes computed in {time.time() - t:.0f}s")
    return out


def prepare(tf: str, htfs: list[str], df1m: pd.DataFrame, out: Outcomes) -> Frame:
    tf_df = data.resample(df1m, tf)
    htf = {label: data.resample(df1m, label) for label in htfs}
    conds = conditions.build(tf_df, htf)
    idx = df1m.index
    pos = np.minimum(idx.searchsorted(tf_df.index), len(idx) - 1)
    ok = (idx[pos] >= tf_df.index) & ((idx[pos] - tf_df.index) <= pd.Timedelta("5min"))
    entry_min = np.where(ok, pos, -1)
    resolved = ok & (out.exit_idx[0, pos] >= 0) & (out.exit_idx[1, pos] >= 0)
    return Frame(tf, tf_df.index, conds, entry_min, resolved,
                 resolved & (out.result[0, pos] == WIN), resolved & (out.result[1, pos] == WIN))


def period_mask(index: pd.DatetimeIndex, period: str) -> np.ndarray:
    a, b = PERIODS[period]
    return np.asarray((index >= pd.Timestamp(a, tz="UTC")) & (index < pd.Timestamp(b, tz="UTC")))


def minute_bounds(idx: pd.DatetimeIndex, period: str) -> tuple[int, int]:
    a, b = PERIODS[period]
    return int(idx.searchsorted(pd.Timestamp(a, tz="UTC"))), int(idx.searchsorted(pd.Timestamp(b, tz="UTC")))


def run_combo(fr: Frame, row, out: Outcomes, idx: pd.DatetimeIndex, period: str):
    return run_combo_between(fr, row, out, *minute_bounds(idx, period))


def run_combo_between(fr: Frame, row, out: Outcomes, vf: int, vt: int):
    lg, sh = combo_mask(row, fr.conds)
    return simulate(fr.entry_min, lg, sh, out, vf, vt)


def screen_frame(fr: Frame, train: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Pick candidate combinations for one timeframe from the bars in `train`
    (default: the train period)."""
    train = (period_mask(fr.index, "train") if train is None else train) & fr.sample
    combos = enumerate_combos(fr.conds, 3)
    cnt = screen(combos, fr.conds, train, fr.win_long, fr.win_short)
    n_screened = len(combos)

    def pick(rows, c):
        nl, wl, ns, ws = c.T
        ok = (nl >= MIN_SIG_PER_SIDE) & (ns >= MIN_SIG_PER_SIDE)
        lb = np.where(ok, wilson_lb(wl + ws, nl + ns), -1)
        wr = np.where(ok & (nl + ns >= 2 * MIN_SIG_PER_SIDE), (wl + ws) / np.maximum(nl + ns, 1), -1)
        by_lb = np.argsort(-lb)
        return rows[by_lb[lb[by_lb] > 0]], rows[np.argsort(-wr)[:TOP_BY_WR]], lb

    by_lb, by_wr, lb = pick(combos, cnt)
    picked = [by_lb[:TOP_BY_LB], by_wr]

    # beam: extend the strongest combinations with a 4th condition
    base = by_lb[:BEAM_BASE]
    ext = extend_combos(base, fr.conds, 4)
    if len(ext):
        cnt4 = screen(ext, fr.conds, train, fr.win_long, fr.win_short)
        n_screened += len(ext)
        ext_lb, _, _ = pick(ext, cnt4)
        picked.append(ext_lb[:BEAM_KEEP])

    rows = np.concatenate([np.pad(p, ((0, 0), (0, 4 - p.shape[1])), constant_values=-1) for p in picked])
    rows = np.unique(rows, axis=0)
    return rows, n_screened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    ap.add_argument("--out", default="reports")
    ap.add_argument("--data", default="BTCUSDT",
                    help="dataset name in data/: BTCUSDT (Binance futures) or BTCUSD-bitstamp")
    ap.add_argument("--null-seed", type=int, default=None,
                    help="run on sign-randomized data (no real edge) to measure selection bias")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df1m = data.load(args.data)
    tag = args.data
    if args.null_seed is not None:
        df1m = data.sign_randomized(df1m, args.null_seed)
        tag = f"{args.data}-null{args.null_seed}"
    idx = df1m.index
    print(f"[{tag}] 1m bars: {len(df1m):,}  {idx[0]} .. {idx[-1]}")
    out = load_outcomes(df1m, tag)

    baseline = {}
    for p in PERIODS:
        vf, vt = minute_bounds(idx, p)
        res = out.result[:, vf:vt]
        resolved = out.exit_idx[:, vf:vt] >= 0
        baseline[p] = {
            "long_wr": float((res[0][resolved[0]] == WIN).mean()),
            "short_wr": float((res[1][resolved[1]] == WIN).mean()),
            "minutes": int(vt - vf),
        }
    print("baseline (enter every minute):", json.dumps(baseline, indent=1))

    frames: dict[str, Frame] = {}
    rows_all = []
    total_screened = 0
    for tf in args.timeframes.split(","):
        t = time.time()
        fr = prepare(tf, TIMEFRAMES[tf], df1m, out)
        frames[tf] = fr
        rows, n_screened = screen_frame(fr)
        total_screened += n_screened
        for row in rows:
            tr = run_combo(fr, row, out, idx, "train")
            s = metrics.summarize(tr, idx)
            if s["trades"] == 0:
                continue
            rows_all.append({"tf": tf, "combo": tuple(int(i) for i in row if i >= 0),
                             "name": combo_name(row, fr.conds),
                             "families": sorted({fr.conds[int(i)].family for i in row if i >= 0}),
                             **{f"train_{k}": v for k, v in s.items()}})
        print(f"[{tf}] {len(fr.conds)} conditions, {n_screened:,} combos screened, "
              f"{len(rows)} simulated in {time.time() - t:.0f}s")

    cand = pd.DataFrame(rows_all)
    cand.to_csv(out_dir / "candidates_train.csv", index=False)

    # the high-WR tail: what the best raw train win rates look like out of sample
    tail = cand[cand["train_trades"] >= 30].sort_values("train_win_rate", ascending=False).head(15).copy()

    ok = ((cand["train_trades"] >= MIN_TRAIN_TRADES)
          & (cand["train_long_trades"] >= MIN_SIDE_TRADES)
          & (cand["train_short_trades"] >= MIN_SIDE_TRADES))
    val = cand[ok].sort_values("train_wr_ci_low", ascending=False).head(TO_VALIDATION).copy()

    def attach(df, period):
        recs = []
        for _, r in df.iterrows():
            fr = frames[r["tf"]]
            s = metrics.summarize(run_combo(fr, list(r["combo"]), out, idx, period), idx)
            if s["trades"]:
                p0 = (s["long_trades"] * baseline[period]["long_wr"]
                      + s["short_trades"] * baseline[period]["short_wr"]) / s["trades"]
                s["p_value"] = metrics.binom_p(s["wins"], s["trades"], p0)
            recs.append({f"{period}_{k}": v for k, v in s.items()})
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)

    val = attach(val, "val")
    val["pooled_trades"] = val["train_trades"] + val["val_trades"].fillna(0)
    val["pooled_wins"] = val["train_wins"] + val["val_wins"].fillna(0)
    val["pooled_wr"] = val["pooled_wins"] / val["pooled_trades"]
    val["pooled_wr_ci_low"] = wilson_lb(val["pooled_wins"], val["pooled_trades"])
    val["val_significant"] = val["val_p_value"] < 0.05
    val.to_csv(out_dir / "candidates_validation.csv", index=False)

    passed = val[(val["val_trades"] >= MIN_VAL_TRADES) & (val["val_exp_net_pct"] > 0)]
    passed = passed.sort_values(["val_significant", "pooled_wr_ci_low"], ascending=False)

    top = []
    for _, r in passed.iterrows():
        fam = set(r["families"])
        if any(t["tf"] == r["tf"] and len(fam & set(t["families"])) / len(fam | set(t["families"])) >= 0.5
               for t in top):
            continue
        top.append(r)
        if len(top) == 5:
            break
    top = pd.DataFrame(top).reset_index(drop=True)
    top = attach(top, "test") if len(top) else top
    tail = attach(attach(tail, "val"), "test")

    yearly = {}
    for i, r in top.iterrows():
        fr = frames[r["tf"]]
        lg, sh = combo_mask(list(r["combo"]), fr.conds)
        tr = simulate(fr.entry_min, lg, sh, out, minute_bounds(idx, "train")[0], len(idx))
        yearly[i] = metrics.yearly(tr, idx)

    top.to_csv(out_dir / "top5.csv", index=False)
    tail.to_csv(out_dir / "high_wr_tail.csv", index=False)
    spec = [{"rank": i + 1, "timeframe": r["tf"],
             "conditions": [frames[r["tf"]].conds[c].name for c in r["combo"]],
             "take_profit_pct": TP * 100, "stop_loss_pct": SL * 100} for i, r in top.iterrows()]
    (out_dir / "top5.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "dataset": tag,
        "data_from": str(idx[0]), "data_to": str(idx[-1]), "bars_1m": len(idx),
        "periods": PERIODS, "timeframes": args.timeframes.split(","),
        "combinations_screened": int(total_screened), "candidates_simulated": len(cand),
        "baseline": baseline,
        "yearly": {int(i): y.reset_index().to_dict("records") for i, y in yearly.items()},
    }, indent=2, default=str))
    print(f"\nscreened {total_screened:,} combinations; top 5:")
    cols = ["tf", "name", "train_trades", "train_win_rate", "val_trades", "val_win_rate", "val_p_value",
            "test_trades", "test_win_rate", "test_p_value", "test_exp_net_pct"]
    print(top[cols].to_string() if len(top) else "(none passed validation)")


if __name__ == "__main__":
    main()
