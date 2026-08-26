import pandas as pd
import pytest

from src.backtester import Backtester
from src.performance_analyzer import PerformanceAnalyzer


def test_costs_are_charged_without_slippage_and_flow_to_equity_metrics():
    prices = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "close": [100, 120, 90],
    })
    backtester = Backtester(slippage_rate=0)

    with_cost = backtester.buy_and_hold(prices, include_cost=True)
    without_cost = backtester.buy_and_hold(prices, include_cost=False)

    shares = 1_000_000 / (100 * (1 + Backtester.COMMISSION_RATE))
    expected_final = shares * 90 * (1 - Backtester.COMMISSION_RATE - Backtester.TAX_RATE)
    assert with_cost["equity_curve"].iloc[-1]["close"] == pytest.approx(expected_final)
    assert with_cost["total_return"] == pytest.approx(
        (expected_final / 1_000_000 - 1) * 100, abs=0.01
    )
    assert with_cost["equity_curve"].iloc[-1]["close"] < without_cost["equity_curve"].iloc[-1]["close"]

    analyzer = PerformanceAnalyzer()
    cost_sharpe = analyzer.sharpe_ratio(with_cost["equity_curve"].copy())["sharpe_ratio"]
    no_cost_sharpe = analyzer.sharpe_ratio(without_cost["equity_curve"].copy())["sharpe_ratio"]
    cost_drawdown = analyzer.max_drawdown(with_cost["equity_curve"])["max_drawdown"]
    no_cost_drawdown = analyzer.max_drawdown(without_cost["equity_curve"])["max_drawdown"]

    assert cost_sharpe != no_cost_sharpe
    assert cost_drawdown > no_cost_drawdown
