"""Download BTCUSDT USDT-M futures 1m klines and store data/BTCUSDT_1m.parquet.

    python -m research.fetch_data [--start 2019-10]
"""
from __future__ import annotations

import argparse

from btcbot import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2019-10", help="first month, YYYY-MM")
    args = ap.parse_args()

    df = data.download(args.symbol, "1m", start=args.start)
    path = data.save(df, args.symbol, "1m")
    gaps = data.gap_report(df)
    print(f"saved {len(df):,} bars {df.index[0]} .. {df.index[-1]} -> {path}")
    print(f"gaps: {len(gaps)} (missing bars total {int(gaps['missing_bars'].sum()) if len(gaps) else 0})")
    if len(gaps):
        print(gaps.sort_values("missing_bars", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
