import pandas as pd
import pytest

from src.out_of_sample import OutOfSampleValidator


def test_evaluate_splits_periods_and_resets_the_account():
    prices = pd.DataFrame({
        "date": ["2023-01-02", "2023-12-29", "2024-01-02", "2024-12-31"],
        "close": [100, 110, 200, 220],
    })

    result = OutOfSampleValidator().evaluate("buy_and_hold", prices)

    assert result["periods"]["in_sample"]["result"]["buy_price"] == 100
    assert result["periods"]["out_of_sample"]["result"]["buy_price"] == 200
    assert result["periods"]["in_sample"]["metrics"]["total_return"] == pytest.approx(10.0)
    assert result["periods"]["out_of_sample"]["metrics"]["total_return"] == pytest.approx(10.0)
    assert list(result["comparison"]["period"]) == ["in_sample", "out_of_sample"]


def test_evaluate_rejects_overlapping_periods():
    prices = pd.DataFrame({"date": ["2023-01-02", "2024-01-02"], "close": [100, 110]})

    with pytest.raises(ValueError, match="must not overlap"):
        OutOfSampleValidator().evaluate(
            "buy_and_hold",
            prices,
            in_sample=("2023-01-01", "2023-12-31"),
            out_of_sample=("2023-12-31", "2024-12-31"),
        )


def test_evaluate_maps_ma_to_the_backtester_strategy_method():
    prices = pd.DataFrame({
        "date": [
            "2023-01-02", "2023-01-03", "2023-01-04",
            "2024-01-02", "2024-01-03", "2024-01-04",
        ],
        "open": [10, 10, 10, 10, 10, 10],
        "close": [10, 10, 10, 10, 10, 10],
    })

    report = OutOfSampleValidator().evaluate(
        "ma",
        prices,
        parameters={"short_window": 2, "long_window": 3},
    )

    assert list(report["comparison"]["period"]) == ["in_sample", "out_of_sample"]
