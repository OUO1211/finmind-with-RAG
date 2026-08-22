import pandas as pd
import pytest

from src.backtester import Backtester


def _prices(buy_signal, sell_signal, opens, closes):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(opens), freq="D"),
        "open": opens,
        "close": closes,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    })


def test_equal_weight_strategy_limits_holdings_and_splits_cash_equally():
    data = {
        "BBB": _prices([True, False, False], [False] * 3, [20, 20, 20], [20, 20, 20]),
        "AAA": _prices([True, False, False], [False] * 3, [10, 10, 10], [10, 10, 10]),
        "CCC": _prices([True, False, False], [False] * 3, [5, 5, 5], [5, 5, 5]),
    }

    result = Backtester().equal_weight_strategy(
        data, max_positions=2, include_cost=False
    )

    # Signals from day one execute at the day-two open.  Ties are resolved by
    # code, so AAA and BBB fill the two available slots while CCC is skipped.
    assert result["open_positions"] == ["AAA", "BBB"]
    assert result["target_weight"] == 0.5
    assert result["equity_curve"].iloc[-1]["close"] == pytest.approx(1_000_000)


def test_equal_weight_strategy_reuses_sale_proceeds_for_a_vacant_slot():
    data = {
        "AAA": _prices(
            [True, False, False, False], [False, True, False, False],
            [10, 10, 15, 15], [10, 15, 15, 15],
        ),
        "BBB": _prices(
            [False, False, True, False], [False] * 4,
            [20, 20, 20, 20], [20, 20, 20, 20],
        ),
    }

    result = Backtester().equal_weight_strategy(
        data, max_positions=1, include_cost=False
    )

    assert result["trades"][0]["code"] == "AAA"
    assert result["trades"][0]["capital_used"] == pytest.approx(1_000_000)
    # AAA is sold for 1.5m on day three; BBB's day-three signal executes on
    # day four and uses every dollar now available in the sole vacant slot.
    assert result["open_positions"] == ["BBB"]
    assert result["equity_curve"].iloc[-1]["close"] == pytest.approx(1_500_000)


@pytest.mark.parametrize("max_positions", [0, -1, 1.5, True])
def test_equal_weight_strategy_rejects_invalid_position_limit(max_positions):
    data = {"AAA": _prices([False], [False], [10], [10])}

    with pytest.raises(ValueError, match="max_positions"):
        Backtester().equal_weight_strategy(data, max_positions=max_positions)
