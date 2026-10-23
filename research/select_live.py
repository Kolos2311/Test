"""Choose the strategies for the bots and write bots/bot1..5.toml.

Applies the selection procedure that was evaluated walk-forward (24 months of
training, ranking by the worst half-year win rate) to the most recent 24
months, then records each pick's history for reference.

    python -m research.select_live --data BTCUSD-bitstamp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btcbot import data, metrics
from btcbot.engine import WIN

from . import run_research as rr
from .walk_forward import candidates, pick_top

TEMPLATE = """# {title}
# {verdict}
# Picked by research/select_live.py on {picked} from data {train_from} .. {train_to}.
# Training-window win rate is optimistic (selection bias); the walk-forward
# out-of-sample win rate of this procedure was {wf_wr:.1%} (see reports/RESULTS.md).
name = "{name}"
mode = "paper"            # paper -> testnet -> mainnet (mainnet also needs HL_CONFIRM_MAINNET=yes)
coin = "BTC"
timeframe = "{tf}"
conditions = {conditions}
take_profit_pct = 1.0
stop_loss_pct = 0.25
notional_usd = 20.0       # position value in USD (Hyperliquid minimum order is 10 USD)
leverage = 5
isolated = true
max_notional_usd = 200.0
entry_slippage_pct = 0.3
max_daily_loss_usd = 5.0
max_consecutive_losses = 12
subaccount_address = ""   # testnet/mainnet with several bots: one sub-account per bot ...
secret_env = "HL_SECRET_KEY"          # ... or a separate wallet per bot: point these to its own variables
account_env = "HL_ACCOUNT_ADDRESS"
state_dir = "state"

[backtest]
train_from = "{train_from}"
train_to = "{train_to}"
train_trades = {trades}
train_win_rate = {wr:.4f}
train_half_year_win_rates = {halves}
history_by_year = {yearly}
pre_sample = {pre}
walk_forward_oos_win_rate = {wf_wr:.4f}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="BTCUSDT")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--score", default="consistency", choices=["consistency", "lb"])
    ap.add_argument("--wf-summary", type=Path, default=Path("reports/walk_forward_t24_m40/summary_BTCUSD-bitstamp.json"))
    ap.add_argument("--out", type=Path, default=Path("bots"))
    ap.add_argument("--null-seed", type=int, default=None,
                    help="control run on sign-randomized data: writes reports only, no bot configs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rr.TOP_BY_LB, rr.TOP_BY_WR, rr.BEAM_KEEP = 800, 200, 300

    df1m = data.load(args.data)
    tag = args.data
    if args.null_seed is not None:
        df1m = data.sign_randomized(df1m, args.null_seed)
        tag = f"{args.data}-null{args.null_seed}"
    idx = df1m.index
    out = rr.load_outcomes(df1m, tag)
    frames = {tf: rr.prepare(tf, rr.TIMEFRAMES[tf], df1m, out) for tf in rr.TIMEFRAMES}
    end = idx[-1].floor("D")
    start = end - pd.DateOffset(months=args.train_months)
    cand = candidates(frames, out, idx, start, end)
    top = pick_top(cand, args.score)
    wf_wr = json.loads(args.wf_summary.read_text())[args.score]["win_rate"] if args.wf_summary.exists() else np.nan

    first = int(idx.searchsorted(pd.Timestamp("2020-01-01", tz="UTC")))
    summary = []
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        tr = rr.run_combo_between(frames[r["tf"]], list(r["combo"]), out, first, len(idx))
        y = idx[tr.entry_min].year
        win = tr.result == WIN
        # before the selection window: data the selection never saw
        pre_m = tr.entry_min < int(idx.searchsorted(start))
        resolved = out.exit_idx[:, first:int(idx.searchsorted(start))] >= 0
        res = out.result[:, first:int(idx.searchsorted(start))]
        base = float(np.concatenate([res[0][resolved[0]], res[1][resolved[1]]]).__eq__(WIN).mean())
        n_pre, w_pre = int(pre_m.sum()), int(win[pre_m].sum())
        pre = {"trades": n_pre, "win_rate": round(w_pre / max(n_pre, 1), 4), "baseline_wr": round(base, 4),
               "p_value": round(metrics.binom_p(w_pre, n_pre, base), 4),
               "exp_net_pct": round(float((tr.ret[pre_m] * 100 - metrics.FEE_TAKER_RT).mean()), 4)}
        yearly = {str(k): [int((y == k).sum()), round(float(win[y == k].mean()), 3)] for k in np.unique(y)}
        name = f"bot{i}"
        cond_toml = "[" + ", ".join(json.dumps(c, ensure_ascii=False) for c in r["conditions"]) + "]"
        yearly_toml = "{ " + ", ".join(f'"{k}" = [{v[0]}, {v[1]}]' for k, v in yearly.items()) + " }"
        text = TEMPLATE.format(
            title=f"{name}: {r['tf']} {r['name']}", picked=pd.Timestamp.now(tz='UTC').date(),
            train_from=start.date(), train_to=end.date(), name=name, tf=r["tf"], conditions=cond_toml,
            trades=r["train_trades"], wr=r["train_wr"],
            halves="[" + ", ".join(f"{h:.3f}" for h in r["train_half_wr"]) + "]",
            yearly=yearly_toml, wf_wr=wf_wr,
            pre="{ " + ", ".join(f"{k} = {v}" for k, v in pre.items()) + " }",
            verdict=(f"CONFIRMED before the selection window: WR {pre['win_rate']:.1%} on {pre['trades']} trades, "
                     f"p = {pre['p_value']} (baseline {pre['baseline_wr']:.1%})" if pre["p_value"] < 0.05 else
                     f"NOT confirmed before the selection window: WR {pre['win_rate']:.1%} on {pre['trades']} "
                     f"trades, p = {pre['p_value']}; keep it on paper"))
        if args.null_seed is None:
            (args.out / f"{name}.toml").write_text(text)
        summary.append({"bot": name, "tf": r["tf"], "conditions": r["conditions"],
                        "train_trades": int(r["train_trades"]), "train_wr": round(float(r["train_wr"]), 4),
                        "train_half_wr": [round(h, 3) for h in r["train_half_wr"]],
                        "history_by_year": yearly, "pre_sample": pre,
                        "history_net_pct_per_trade": round(float((tr.ret * 100 - metrics.FEE_TAKER_RT).mean()), 4)})
    Path("reports").mkdir(exist_ok=True)
    suffix = "" if args.null_seed is None else f"_null{args.null_seed}"
    Path(f"reports/live_selection{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
