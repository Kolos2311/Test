"""Run one or several bots.

    python -m btcbot.live bots/bot1.toml bots/bot2.toml ...
    python -m btcbot.live bots/*.toml --check        # validate configs and data, no trading
    python -m btcbot.live bots/bot3.toml --reset-risk

Market data (signals) always come from Hyperliquid mainnet candles; orders go to
the venue chosen by each bot's `mode` (paper = simulated, testnet, mainnet).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

from . import config as cfgmod
from .bot import Bot
from .broker import HyperliquidBroker, PaperBroker
from .market import MAINNET_API_URL, TESTNET_API_URL, MarketData, PublicClient
from .signals import available_conditions, evaluate


def make_broker(cfg: cfgmod.BotConfig, market: MarketData):
    if cfg.mode == "paper":
        return PaperBroker(cfg.coin, market, cfg.state_path / "paper.json")
    sec = cfgmod.Secrets.from_env(cfg.secret_env, cfg.account_env)
    url = MAINNET_API_URL if cfg.mode == "mainnet" else TESTNET_API_URL
    broker = HyperliquidBroker.connect(cfg.coin, url, sec.secret_key, sec.account_address,
                                       cfg.subaccount_address, cfg.entry_slippage_pct / 100)
    broker.setup(cfg.leverage, cfg.isolated)
    return broker


def check_accounts(cfgs: list[cfgmod.BotConfig]) -> None:
    """Hyperliquid nets positions per coin per account: two bots must never share one."""
    seen = {}
    for c in cfgs:
        if c.mode == "paper":
            continue
        account = c.subaccount_address or os.environ.get(c.account_env, c.account_env)
        key = (c.mode, account.lower(), c.coin)
        if key in seen:
            raise SystemExit(f"bots {seen[key]} and {c.name} would trade the same {c.coin} position on "
                             f"{key[1]} ({c.mode}); give each bot its own sub-account")
        seen[key] = c.name


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m btcbot.live")
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true", help="validate configs, print current signals, exit")
    ap.add_argument("--reset-risk", action="store_true", help="clear the consecutive-loss pause")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between exchange syncs")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("bots.log")])
    cfgs = [cfgmod.load(p) for p in args.configs]
    names = [c.name for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("bot names must be unique")
    check_accounts(cfgs)

    market = MarketData(PublicClient(MAINNET_API_URL))
    if args.check:
        for c in cfgs:
            frames = market.frames(c.timeframe, cfgmod.HTF_FOR[c.timeframe])
            unknown = set(c.conditions) - set(available_conditions(c.timeframe, frames))
            if unknown:
                raise SystemExit(f"{c.name}: unknown conditions {sorted(unknown)}")
            sig = evaluate(c.timeframe, c.conditions, frames)
            print(f"{c.name:8s} {c.mode:8s} {c.timeframe:6s} last bar {sig.bar_close}  "
                  f"long={sig.long} short={sig.short}  conditions: {' + '.join(c.conditions)}")
        return

    bots = [Bot(c, make_broker(c, market), market) for c in cfgs]
    if args.reset_risk:
        for b in bots:
            b.reset_risk()
    threads = [threading.Thread(target=b.run, kwargs={"poll_s": args.poll}, name=b.cfg.name, daemon=True)
               for b in bots]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logging.warning("interrupted: bots stop, open positions keep their exchange-side TP/SL orders")


if __name__ == "__main__":
    main()
