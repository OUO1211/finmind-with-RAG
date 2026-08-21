import pandas as pd
import pytest

from src.backtester import Backtester


def test_fixed_slippage_applies_adverse_fill_prices_and_costs():
    prices = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'close': [100, 110],
    })

    result = Backtester(slippage_rate=0.001).buy_and_hold(prices)

    # Buy: 100 * 1.001 * 1.001425; sell: 110 * .999 * (1 - .001425 - .003)
    expected = (
        110 * 0.999 * (1 - 0.001425 - 0.003)
        / (100 * 1.001 * (1 + 0.001425))
        - 1
    ) * 100
    assert result['returns'] == pytest.approx(expected, abs=0.01)
    assert result['slippage_rate'] == 0.001


def test_trade_records_keep_market_price_and_expose_fill_prices():
    prices = pd.DataFrame({
        'date': [f'2024-01-{day:02d}' for day in range(1, 11)],
        'open': [9, 9, 9, 18, 19, 19, 6, 5, 5, 5],
        'close': [10, 10, 10, 20, 20, 20, 5, 5, 5, 5],
    })

    trade = Backtester(slippage_rate=0.002).ma_strategy(
        prices, short_window=2, long_window=3
    )['trades'][0]

    assert trade['buy_price'] == 19
    assert trade['sell_price'] == 5
    assert trade['buy_fill_price'] == pytest.approx(19.038)
    assert trade['sell_fill_price'] == pytest.approx(4.99)


def test_negative_slippage_rate_is_rejected():
    with pytest.raises(ValueError, match='non-negative'):
        Backtester(slippage_rate=-0.001)
