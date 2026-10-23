"""Build data/BTCUSD-bitstamp_1m.parquet from a local clone of
github.com/ff137/bitstamp-btcusd-minute-data (used when Binance archives are
unreachable).

    python -m research.import_bitstamp /path/to/bitstamp-btcusd-minute-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

from btcbot import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--start", default="2019-10-01")
    args = ap.parse_args()
    df = data.load_bitstamp(args.repo, args.start)
    path = data.save(df, "BTCUSD-bitstamp", "1m")
    zero = (df["volume"] == 0).mean()
    print(f"saved {len(df):,} bars {df.index[0]} .. {df.index[-1]} -> {path}")
    print(f"zero-volume minutes: {zero:.2%}, clipped isolated wicks: {df.attrs['clipped_wicks']}")


if __name__ == "__main__":
    main()
