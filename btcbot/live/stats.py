"""Live / paper results per bot compared with the backtest.

    python -m btcbot.live.stats [--state state] [--bots bots]
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from ..metrics import wilson


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m btcbot.live.stats")
    ap.add_argument("--state", type=Path, default=Path("state"))
    ap.add_argument("--bots", type=Path, default=Path("bots"))
    args = ap.parse_args(argv)

    rows = []
    for cfg_path in sorted(args.bots.glob("*.toml")):
        cfg = tomllib.loads(cfg_path.read_text())
        journal = args.state / cfg["name"] / "trades.csv"
        df = pd.read_csv(journal) if journal.exists() else pd.DataFrame(columns=["result", "net_usd"])
        n, wins = len(df), int((df["result"] == "win").sum())
        lo, hi = wilson(wins, n) if n else (float("nan"), float("nan"))
        pre = cfg.get("backtest", {}).get("pre_sample", {})
        rows.append({"bot": cfg["name"], "mode": cfg.get("mode"), "tf": cfg["timeframe"], "trades": n,
                     "win_rate": wins / n if n else float("nan"), "wr_95ci": f"{lo:.0%}..{hi:.0%}" if n else "-",
                     "net_usd": round(float(df["net_usd"].sum()), 2) if n else 0.0,
                     "backtest_wr": pre.get("win_rate"), "breakeven_wr": 0.27})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\nbacktest_wr: win rate before the selection window (data the selection never saw).\n"
          "If backtest_wr is above the live 95% interval after ~50+ trades, the edge is likely gone.")


if __name__ == "__main__":
    main()
