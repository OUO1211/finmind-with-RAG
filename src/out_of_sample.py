"""In-sample / out-of-sample validation for the backtesting strategies.

The validator deliberately runs each period as a separate backtest.  This
means an open in-sample position is not carried into the out-of-sample
period, and no out-of-sample observation can influence parameter selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.backtester import Backtester
from src.performance_analyzer import PerformanceAnalyzer


class OutOfSampleValidator:
    """Evaluate one fixed strategy on independent IS and OOS date ranges."""

    DEFAULT_IN_SAMPLE = ("2019-01-01", "2023-12-31")
    DEFAULT_OUT_OF_SAMPLE = ("2024-01-01", "2026-12-31")

    _STRATEGIES = {"buy_and_hold", "ma", "rsi", "kd", "pe"}
    _BACKTESTER_METHODS = {
        "buy_and_hold": "buy_and_hold",
        "ma": "ma_strategy",
        "rsi": "rsi_strategy",
        "kd": "kd_strategy",
        "pe": "pe_strategy",
    }

    def __init__(
        self,
        backtester: Backtester | None = None,
        performance_analyzer: PerformanceAnalyzer | None = None,
    ) -> None:
        self.backtester = backtester or Backtester()
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer()

    
    def evaluate(
        self,
        strategy_name: str,
        price_df: pd.DataFrame,
        *,
        parameters: Mapping[str, Any] | None = None,
        per_df: pd.DataFrame | None = None,
        in_sample: tuple[str, str] = DEFAULT_IN_SAMPLE,
        out_of_sample: tuple[str, str] = DEFAULT_OUT_OF_SAMPLE,
    ) -> dict[str, Any]:
        """Run a previously selected strategy without changing its parameters.

        ``parameters`` must be chosen using only the in-sample period. Both
        intervals are inclusive and each interval starts with a fresh account.
        """
        if strategy_name not in self._STRATEGIES:
            supported = ", ".join(sorted(self._STRATEGIES))
            raise ValueError(f"Unsupported strategy {strategy_name!r}; use one of: {supported}")
        self._validate_periods(in_sample, out_of_sample)
        parameters = dict(parameters or {})
        price = self._prepare_frame(price_df, "price_df")
        per = self._prepare_frame(per_df, "per_df") if per_df is not None else None

        periods: dict[str, dict[str, Any]] = {}
        for label, date_range in (("in_sample", in_sample), ("out_of_sample", out_of_sample)):
            start_date, end_date = date_range
            period_price = self._slice(price, start_date, end_date)
            if period_price.empty:
                raise ValueError(f"No price data in {label} period {start_date} to {end_date}")
            period_per = self._slice(per, start_date, end_date) if per is not None else None
            result = self._run(strategy_name, period_price, period_per, parameters)
            periods[label] = {
                "start_date": start_date,
                "end_date": end_date,
                "result": result,
                "metrics": self._metrics(result),
            }

        comparison = pd.DataFrame(
            [{"period": label, **value["metrics"]} for label, value in periods.items()]
        )
        return {
            "strategy_name": strategy_name,
            "parameters": parameters,
            "periods": periods,
            "comparison": comparison,
        }

    
    @staticmethod
    def _prepare_frame(frame: pd.DataFrame | None, name: str) -> pd.DataFrame:
        """ 依照時間整理dataframe """
        if frame is None:
            raise ValueError(f"{name} is required")
        if "date" not in frame.columns:
            raise ValueError(f"{name} must contain a 'date' column")
        prepared = frame.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])
        return prepared.sort_values("date").drop_duplicates("date", keep="last")

    
    @staticmethod
    def _slice(frame: pd.DataFrame | None, start_date: str, end_date: str) -> pd.DataFrame | None:
        """ 找出位於start_date / end_date 之間row的索引 """
        if frame is None:
            return None
        return frame.loc[frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()

    
    @staticmethod
    def _validate_periods(in_sample: tuple[str, str], out_of_sample: tuple[str, str]) -> None:
        """ 檢驗時間合理性 """
        in_start, in_end = map(pd.Timestamp, in_sample)
        out_start, out_end = map(pd.Timestamp, out_of_sample)
        if in_start > in_end or out_start > out_end:
            raise ValueError("Each validation period must start on or before it ends")
        if in_end >= out_start:
            raise ValueError("In-sample and out-of-sample periods must not overlap")

    def _run(
        self,
        strategy_name: str,
        price_df: pd.DataFrame,
        per_df: pd.DataFrame | None,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """ 執行回測方法 """
        if strategy_name == "pe":
            if per_df is None:
                raise ValueError("per_df is required for the 'pe' strategy")
            return self.backtester.pe_strategy(price_df, per_df, **parameters)
        method_name = self._BACKTESTER_METHODS[strategy_name]
        return getattr(self.backtester, method_name)(price_df, **parameters)

    
    def _metrics(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """ 回傳各項結果 """
        curve = result["equity_curve"].copy()
        curve["date"] = pd.to_datetime(curve["date"])
        curve = curve.sort_values("date").reset_index(drop=True)
        initial_value = float(curve.iloc[0]["close"])
        final_value = float(curve.iloc[-1]["close"])
        total_return = (final_value / initial_value - 1) * 100
        elapsed_days = max((curve.iloc[-1]["date"] - curve.iloc[0]["date"]).days, 1)
        annualized_return = ((final_value / initial_value) ** (365.25 / elapsed_days) - 1) * 100
        sharpe = self.performance_analyzer.sharpe_ratio(curve.copy())["sharpe_ratio"]
        return {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": self.performance_analyzer.max_drawdown(curve)["max_drawdown"],
            "sharpe_ratio": sharpe,
            "total_trades": result.get("total_trades", 0),
        }
