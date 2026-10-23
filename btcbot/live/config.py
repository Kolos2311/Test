"""Bot configuration (TOML file per bot) and secrets (environment variables).

Secrets never live in config files:
  HL_SECRET_KEY        private key of the Hyperliquid API wallet (agent) that signs orders
  HL_ACCOUNT_ADDRESS   address of the main account that approved the API wallet
  HL_CONFIRM_MAINNET   must be "yes" for any bot with mode = "mainnet"
A bot can read its keys from other variables (`secret_env`, `account_env`), e.g.
to give every bot its own wallet instead of a sub-account.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("paper", "testnet", "mainnet")
TIMEFRAMES = ("5min", "15min", "30min", "1h", "4h")
# higher timeframes joined to each base timeframe (same as in the research)
HTF_FOR = {
    "5min": ["1h", "4h", "1D"],
    "15min": ["1h", "4h", "1D"],
    "30min": ["4h", "1D"],
    "1h": ["4h", "1D"],
    "4h": ["1D"],
}


@dataclass
class BotConfig:
    name: str
    timeframe: str
    conditions: list[str]
    mode: str = "paper"
    coin: str = "BTC"
    take_profit_pct: float = 1.0
    stop_loss_pct: float = 0.25
    notional_usd: float = 20.0          # position size (price * size), not margin
    leverage: int = 5
    isolated: bool = True
    max_notional_usd: float = 1000.0    # hard cap, protects against config typos
    entry_slippage_pct: float = 0.3     # worst accepted price for the market entry
    max_daily_loss_usd: float = 10.0    # stop opening trades for the rest of the UTC day
    max_consecutive_losses: int = 10    # pause until restarted with --reset-risk
    subaccount_address: str = ""        # trade this sub-account (one per bot on a real account)
    secret_env: str = "HL_SECRET_KEY"
    account_env: str = "HL_ACCOUNT_ADDRESS"
    state_dir: str = "state"
    backtest: dict = field(default_factory=dict)  # reference stats, informational only

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: mode must be one of {MODES}")
        if self.timeframe not in TIMEFRAMES:
            raise ValueError(f"{self.name}: timeframe must be one of {TIMEFRAMES}")
        if not self.conditions:
            raise ValueError(f"{self.name}: no conditions")
        if not 0 < self.stop_loss_pct < 5 or not 0 < self.take_profit_pct < 20:
            raise ValueError(f"{self.name}: take_profit_pct / stop_loss_pct out of range")
        if not 10 <= self.notional_usd <= self.max_notional_usd:
            raise ValueError(f"{self.name}: notional_usd must be in [10, max_notional_usd] "
                             "(Hyperliquid minimum order value is 10 USD)")
        if not 1 <= self.leverage <= 40:
            raise ValueError(f"{self.name}: leverage must be 1..40")
        if self.mode == "mainnet" and os.environ.get("HL_CONFIRM_MAINNET") != "yes":
            raise ValueError(f"{self.name}: mode=mainnet requires HL_CONFIRM_MAINNET=yes in the environment")

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / self.name


def load(path: str | Path) -> BotConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = BotConfig(**raw)
    cfg.validate()
    return cfg


@dataclass
class Secrets:
    secret_key: str
    account_address: str

    @classmethod
    def from_env(cls, secret_env: str = "HL_SECRET_KEY", account_env: str = "HL_ACCOUNT_ADDRESS") -> "Secrets":
        key = os.environ.get(secret_env, "")
        addr = os.environ.get(account_env, "")
        if not key or not addr:
            raise RuntimeError(f"set {secret_env} and {account_env} for testnet/mainnet modes")
        return cls(key, addr)
