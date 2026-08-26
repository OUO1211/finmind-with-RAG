"""Loading and validation for immutable forward-test specifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .backtester import Backtester


@dataclass(frozen=True)
class ForwardTestSpec:
    """The complete, versioned specification used by a forward test."""

    version: str
    frozen_at: date
    start_date: date
    initial_capital: float
    universe_max_stocks: int
    kd_period: int
    max_positions: int
    commission_rate: float
    sell_tax_rate: float
    slippage_rate: float

    @classmethod
    def load(cls, path: str | Path) -> "ForwardTestSpec":
        with Path(path).open(encoding="utf-8") as source:
            raw: dict[str, Any] = yaml.safe_load(source)

        try:
            strategy = raw["strategy"]
            universe = raw["universe"]
            portfolio = raw["portfolio"]
            costs = raw["costs"]
            spec = cls(
                version=str(raw["version"]),
                frozen_at=_as_date(raw["frozen_at"], "frozen_at"),
                start_date=_as_date(raw["start_date"], "start_date"),
                initial_capital=float(raw["initial_capital"]),
                universe_max_stocks=int(universe["max_stocks"]),
                kd_period=int(strategy["period"]),
                max_positions=int(portfolio["max_positions"]),
                commission_rate=float(costs["commission_rate"]),
                sell_tax_rate=float(costs["sell_tax_rate"]),
                slippage_rate=float(costs["slippage_rate"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid forward-test specification: {path}") from error

        spec._validate(raw)
        return spec

    def make_backtester(self) -> Backtester:
        """Create the cost model dictated by this frozen specification."""
        return Backtester(
            slippage_rate=self.slippage_rate,
            commission_rate=self.commission_rate,
            tax_rate=self.sell_tax_rate,
        )

    def _validate(self, raw: dict[str, Any]) -> None:
        if self.version != "forward-test-v1":
            raise ValueError("Only forward-test-v1 is supported")
        if self.start_date <= self.frozen_at:
            raise ValueError("start_date must be after frozen_at")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.universe_max_stocks != 50 or raw["universe"].get("rule") != "daily_top_market_cap":
            raise ValueError("v1 universe must be daily_top_market_cap with 50 stocks")
        if self.kd_period != 9 or raw["strategy"].get("name") != "kd_crossover":
            raise ValueError("v1 strategy must be kd_crossover with period 9")
        if raw["strategy"].get("signal_time") != "close" or raw["strategy"].get("execution_time") != "next_trading_day_open":
            raise ValueError("v1 signals must execute at the next trading-day open")
        if self.max_positions != 5 or raw["portfolio"].get("allocation") != "equal_weight_vacant_slots":
            raise ValueError("v1 portfolio must use five equal-weight vacant slots")
        if self.commission_rate != 0.001425 or self.sell_tax_rate != 0.003 or self.slippage_rate != 0.001:
            raise ValueError("v1 transaction costs must match the frozen rates")


def _as_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"{field_name} must be an ISO date")
