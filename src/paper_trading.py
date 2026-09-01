"""Persistent daily paper-trading workflow for the frozen forward-test spec.

The caller supplies close-of-day factors and the next session's opening prices.
This keeps live data acquisition separate from order generation and makes every
decision reproducible from the order and fill ledgers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping
import json
import os
import uuid

import pandas as pd

from .forward_test_spec import ForwardTestSpec


ORDER_COLUMNS = [
    "order_id", "signal_date", "execution_date", "code", "side", "status",
    "reason", "market_price", "fill_price", "shares", "capital_used", "proceeds",
]
FILL_COLUMNS = ["order_id", "execution_date", "code", "side", "market_price", "fill_price", "shares", "capital_used", "proceeds"]


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    signal_date: date
    execution_date: date
    code: str
    side: str
    status: str = "pending"
    reason: str = "kd_crossover"


@dataclass(frozen=True)
class PaperFill:
    order_id: str
    execution_date: date
    code: str
    side: str
    market_price: float
    fill_price: float
    shares: float
    capital_used: float = 0.0
    proceeds: float = 0.0


@dataclass
class PaperTradingState:
    cash: float
    positions: dict[str, dict]


class PaperTradingService:
    """Generate close signals and execute their pending orders at the next open."""

    def __init__(self, ledger_dir: str | Path, spec: ForwardTestSpec):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.backtester = spec.make_backtester()
        self.orders_path = self.ledger_dir / "orders.csv"
        self.fills_path = self.ledger_dir / "fills.csv"
        self.state_path = self.ledger_dir / "state.json"
        self._initialize_ledger()

    def generate_close_orders(
        self,
        signal_date: date,
        factor_table: pd.DataFrame,
        universe: Iterable[str],
        trading_days: Iterable[date],
    ) -> list[PaperOrder]:
        """Persist KD crossover orders for the first trading day after ``signal_date``.

        Buy signals are limited to that day's universe; sell signals apply to
        holdings even after a stock leaves the universe.
        """
        execution_date = self._next_trading_day(signal_date, trading_days)
        existing = self._read_orders()
        same_run = existing[existing["signal_date"] == signal_date.isoformat()]
        if not same_run.empty:
            return [self._order_from_row(row) for _, row in same_run.iterrows()]

        signals = self._kd_signals(signal_date, factor_table)
        state = self.load_state()
        universe_codes = {str(code) for code in universe}
        pending_buys = set(existing.loc[(existing["status"] == "pending") & (existing["side"] == "buy"), "code"])
        orders: list[PaperOrder] = []
        for code in sorted(state.positions):
            if signals.get(code) == "sell":
                orders.append(self._new_order(signal_date, execution_date, code, "sell"))
        for code in sorted(universe_codes):
            if code not in state.positions and code not in pending_buys and signals.get(code) == "buy":
                orders.append(self._new_order(signal_date, execution_date, code, "buy"))
        self._append_orders(orders)
        return orders

    def execute_open(self, execution_date: date, opening_prices: Mapping[str, float]) -> list[PaperFill]:
        """Fill due orders, selling first so released cash and slots are reusable."""
        orders = self._read_orders()
        due = orders[(orders["status"] == "pending") & (orders["execution_date"] == execution_date.isoformat())].copy()
        if due.empty:
            return []
        state = self.load_state()
        fills: list[PaperFill] = []
        for side in ("sell", "buy"):
            for index, order in due[due["side"] == side].sort_values("code").iterrows():
                code = str(order["code"])
                if code not in opening_prices:
                    orders.loc[index, "status"] = "cancelled"
                    continue
                price = float(opening_prices[code])
                if price <= 0:
                    raise ValueError(f"opening price for {code} must be positive")
                fill = self._execute_order(order, state, execution_date, price)
                if fill is None:
                    orders.loc[index, "status"] = "cancelled"
                    continue
                fills.append(fill)
                orders.loc[index, ["status", "market_price", "fill_price", "shares", "capital_used", "proceeds"]] = [
                    "executed", fill.market_price, fill.fill_price, fill.shares, fill.capital_used, fill.proceeds,
                ]
        self._write_orders(orders)
        self._append_fills(fills)
        self._write_state(state)
        return fills

    def load_state(self) -> PaperTradingState:
        if not self.state_path.exists():
            return PaperTradingState(cash=self.spec.initial_capital, positions={})
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return PaperTradingState(cash=float(raw["cash"]), positions=dict(raw["positions"]))

    def _execute_order(self, order: pd.Series, state: PaperTradingState, execution_date: date, price: float) -> PaperFill | None:
        code, side = str(order["code"]), str(order["side"])
        if side == "sell":
            position = state.positions.pop(code, None)
            if position is None:
                return None
            shares = float(position["shares"])
            fill_price = self.backtester._fill_price(price, "sell", True)
            proceeds = self.backtester._liquidation_cash(shares, price, True)
            state.cash += proceeds
            return PaperFill(str(order["order_id"]), execution_date, code, side, price, fill_price, shares, proceeds=proceeds)
        if code in state.positions or len(state.positions) >= self.spec.max_positions:
            return None
        vacant_slots = self.spec.max_positions - len(state.positions)
        capital_used = state.cash / vacant_slots
        shares = self.backtester._position_shares(capital_used, price, True)
        if shares <= 0:
            return None
        fill_price = self.backtester._fill_price(price, "buy", True)
        state.cash -= capital_used
        state.positions[code] = {"shares": shares, "capital_used": capital_used, "buy_date": execution_date.isoformat(), "buy_price": price, "buy_fill_price": fill_price}
        return PaperFill(str(order["order_id"]), execution_date, code, side, price, fill_price, shares, capital_used=capital_used)

    def _kd_signals(self, signal_date: date, factor_table: pd.DataFrame) -> dict[str, str]:
        required = {"date", "code", "factor_KD_K", "factor_KD_D"}
        missing = required.difference(factor_table.columns)
        if missing:
            raise ValueError(f"factor_table is missing required columns: {', '.join(sorted(missing))}")
        data = factor_table.copy()
        data["date"] = pd.to_datetime(data["date"]).dt.date
        signals: dict[str, str] = {}
        for code, rows in data.groupby(data["code"].astype(str)):
            rows = rows[rows["date"] <= signal_date].sort_values("date")
            if len(rows) < 2 or rows.iloc[-1]["date"] != signal_date:
                continue
            previous, current = rows.iloc[-2], rows.iloc[-1]
            values = [previous["factor_KD_K"], previous["factor_KD_D"], current["factor_KD_K"], current["factor_KD_D"]]
            if pd.isna(values).any():
                continue
            if previous["factor_KD_K"] < previous["factor_KD_D"] and current["factor_KD_K"] > current["factor_KD_D"]:
                signals[code] = "buy"
            elif previous["factor_KD_K"] > previous["factor_KD_D"] and current["factor_KD_K"] < current["factor_KD_D"]:
                signals[code] = "sell"
        return signals

    def _next_trading_day(self, signal_date: date, trading_days: Iterable[date]) -> date:
        candidates = sorted(day for day in trading_days if day > signal_date)
        if not candidates:
            raise ValueError("a next trading day is required to create paper orders")
        return candidates[0]

    def _new_order(self, signal_date: date, execution_date: date, code: str, side: str) -> PaperOrder:
        return PaperOrder(uuid.uuid4().hex, signal_date, execution_date, str(code), side)

    def _initialize_ledger(self) -> None:
        if not self.orders_path.exists():
            pd.DataFrame(columns=ORDER_COLUMNS).to_csv(self.orders_path, index=False)
        if not self.fills_path.exists():
            pd.DataFrame(columns=FILL_COLUMNS).to_csv(self.fills_path, index=False)

    def _read_orders(self) -> pd.DataFrame:
        return pd.read_csv(self.orders_path, dtype={"order_id": str, "code": str, "side": str, "status": str})

    def _append_orders(self, paper_orders: list[PaperOrder]) -> None:
        if not paper_orders:
            return
        rows = [{**asdict(order), "signal_date": order.signal_date.isoformat(), "execution_date": order.execution_date.isoformat(), "market_price": None, "fill_price": None, "shares": None, "capital_used": None, "proceeds": None} for order in paper_orders]
        new_orders = pd.DataFrame(rows, columns=ORDER_COLUMNS)
        new_orders.to_csv(self.orders_path, mode="a", header=False, index=False)

    def _append_fills(self, fills: list[PaperFill]) -> None:
        if fills:
            new_fills = pd.DataFrame(
                [asdict(fill) for fill in fills], columns=FILL_COLUMNS
            )
            new_fills.to_csv(self.fills_path, mode="a", header=False, index=False)

    def _write_orders(self, orders: pd.DataFrame) -> None:
        orders.to_csv(self.orders_path, index=False)

    def _write_state(self, state: PaperTradingState) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)

    @staticmethod
    def _order_from_row(row: pd.Series) -> PaperOrder:
        return PaperOrder(str(row["order_id"]), date.fromisoformat(row["signal_date"]), date.fromisoformat(row["execution_date"]), str(row["code"]), str(row["side"]), str(row["status"]), str(row["reason"]))
