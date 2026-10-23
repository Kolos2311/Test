"""Evaluate a strategy (timeframe + list of condition names) on the latest
closed bar with exactly the condition code used in the backtest."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import conditions
from .config import HTF_FOR


@dataclass
class Signal:
    bar_close: pd.Timestamp
    long: bool
    short: bool

    @property
    def side(self) -> str | None:
        if self.long == self.short:
            return None
        return "long" if self.long else "short"


def available_conditions(timeframe: str, frames: dict[str, pd.DataFrame]) -> list[str]:
    base = frames[timeframe]
    return [c.name for c in conditions.build(base, {h: frames[h] for h in HTF_FOR[timeframe]})]


def evaluate(timeframe: str, names: list[str], frames: dict[str, pd.DataFrame]) -> Signal:
    base = frames[timeframe]
    conds = {c.name: c for c in conditions.build(base, {h: frames[h] for h in HTF_FOR[timeframe]})}
    missing = [n for n in names if n not in conds]
    if missing:
        raise KeyError(f"unknown conditions for {timeframe}: {missing}")
    long = all(bool(conds[n].long[-1]) for n in names)
    short = all(bool(conds[n].short[-1]) for n in names)
    return Signal(base.index[-1], long, short)
