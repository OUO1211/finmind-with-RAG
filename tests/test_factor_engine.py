import pandas as pd
import pytest

from src.factor_engine import FactorEngine


@pytest.fixture
def price_df():
    return pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"],
        "open": [12, 10, 11, 13, 14],
        "close": [13, 11, 12, 14, 15],
        "min": [11, 9, 10, 12, 13],
        "max": [14, 12, 13, 15, 16],
    })


def test_build_factor_table_aligns_all_factors_by_date(price_df):
    roe_df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-04"],
        "ROE": [8.5, 9.0],
    })

    result = FactorEngine().build_factor_table(
        "2330", price_df, roe_df,
        ma_short_window=2, ma_long_window=3, rsi_period=2, kd_period=2,
    )

    assert list(result.columns) == [
        "date", "code", "open", "close", "factor_ROE",
        "factor_MA_short", "factor_MA_long", "factor_RSI",
        "factor_KD_K", "factor_KD_D",
    ]
    assert list(result["date"]) == [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    ]
    assert set(result["code"]) == {"2330"}
    assert pd.isna(result.loc[0, "factor_ROE"])
    assert result.loc[1, "factor_ROE"] == 8.5
    assert result.loc[2, "factor_ROE"] == 8.5
    assert result.loc[3, "factor_ROE"] == 9.0
    assert result.loc[2, "factor_MA_short"] == pytest.approx(12.5)
    assert result.loc[2, "factor_MA_long"] == pytest.approx(12.0)
    assert result.loc[1, "factor_KD_K"] == pytest.approx(58.333333, rel=1e-6)


def test_build_factor_table_keeps_roe_column_without_fundamental_data(price_df):
    result = FactorEngine().build_factor_table("2330", price_df, kd_period=2)

    assert result["factor_ROE"].isna().all()


def test_build_factor_table_rejects_invalid_price_schema():
    price_df = pd.DataFrame({"date": ["2024-01-01"], "close": [10]})

    with pytest.raises(ValueError, match="max, min, open"):
        FactorEngine().build_factor_table("2330", price_df)
