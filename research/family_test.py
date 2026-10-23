"""Family-level test of the one pattern that kept reappearing in walk-forward
picks: an oversold trigger in the direction of a higher trend (long), mirrored
for shorts. Instead of picking the best variant, every variant of a fixed grid
is evaluated year by year on the full history, on real and on null data.

    python -m research.family_test --data BTCUSD-bitstamp [--null-seed 1]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import data, metrics
from btcbot.engine import WIN, simulate

from . import run_research as rr

TIMEFRAMES = ["5min", "15min", "30min", "1h"]
TRENDS = ["close>EMA200", "EMA20>EMA50>EMA200", "Supertrend(10,3) up",
          "1h close>EMA50", "4h close>EMA50", "4h Supertrend up", "1D close>EMA50"]
TRIGGERS = ["RSI2<10", "RSI2<5", "Stoch K<10", "CCI<-200", "close<BB lower", "MFI<20",
            "z50<-2", "RSI14<30", "Williams%R<-90"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="BTCUSDT")
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--out", default="reports/family_test")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df1m = data.load(args.data)
    tag = args.data
    if args.null_seed is not None:
        df1m = data.sign_randomized(df1m, args.null_seed)
        tag = f"{args.data}-null{args.null_seed}"
    idx = df1m.index
    out = rr.load_outcomes(df1m, tag)
    start = int(idx.searchsorted(pd.Timestamp("2020-01-01", tz="UTC")))
    years = idx.year.to_numpy()

    resolved = out.exit_idx[:, start:] >= 0
    base = float(np.concatenate([out.result[0, start:][resolved[0]], out.result[1, start:][resolved[1]]]) .__eq__(WIN).mean())
    rows = []
    for tf in TIMEFRAMES:
        fr = rr.prepare(tf, rr.TIMEFRAMES[tf], df1m, out)
        by_name = {c.name: c for c in fr.conds}
        for trend, trig in itertools.product(TRENDS, TRIGGERS):
            if trend not in by_name or trig not in by_name:
                continue
            a, b = by_name[trend], by_name[trig]
            tr = simulate(fr.entry_min, a.long & b.long, a.short & b.short, out, start, len(idx))
            if len(tr) < 50:
                continue
            win = tr.result == WIN
            y = years[tr.entry_min]
            yearly = {int(k): float(win[y == k].mean()) for k in np.unique(y) if (y == k).sum() >= 10}
            rows.append({"tf": tf, "trend": trend, "trigger": trig, "trades": len(tr), "win_rate": win.mean(),
                         "exp_net_pct": float((tr.ret * 100 - metrics.FEE_TAKER_RT).mean()),
                         "years_above_base": sum(v > base for v in yearly.values()), "years": len(yearly),
                         "worst_year_wr": min(yearly.values()) if yearly else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"family_{tag}.csv", index=False)
    summary = {
        "dataset": tag, "baseline_wr": base, "variants": len(res),
        "median_wr": float(res["win_rate"].median()),
        "share_above_base": float((res["win_rate"] > base).mean()),
        "share_above_base_every_year": float((res["years_above_base"] == res["years"]).mean()),
        "share_net_positive": float((res["exp_net_pct"] > 0).mean()),
        "best_wr": float(res["win_rate"].max()),
        "pooled_wr": float((res["win_rate"] * res["trades"]).sum() / res["trades"].sum()),
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(res.sort_values("win_rate", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
