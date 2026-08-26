import pandas as pd
import pytest

from src.backtester import Backtester


def test_ma_strategy_uses_only_configured_fraction_of_cash():
    prices = pd.DataFrame({
        'date': [f'2024-01-{day:02d}' for day in range(1, 11)],
        'open': [9, 9, 9, 18, 19, 19, 6, 5, 5, 5],
        'close': [10, 10, 10, 20, 20, 20, 5, 5, 5, 5],
    })

    result = Backtester(position_size=0.2).ma_strategy(
        prices, short_window=2, long_window=3
    )
    trade = result['trades'][0]

    assert result['position_size'] == 0.2
    assert trade['position_size'] == 0.2
    assert trade['capital_used'] == pytest.approx(200_000)
    shares = 200_000 / (19 * (1 + Backtester.COMMISSION_RATE))
    assert trade['shares'] == pytest.approx(shares)
    # 80% cash is preserved; the 20% position is then sold at 5.
    assert result['equity_curve'].iloc[-1]['close'] == pytest.approx(
        800_000 + shares * 5 * (1 - Backtester.COMMISSION_RATE - Backtester.TAX_RATE)
    )


@pytest.mark.parametrize('position_size', [0, -0.1, 1.1])
def test_invalid_position_size_is_rejected(position_size):
    with pytest.raises(ValueError, match='position_size'):
        Backtester(position_size=position_size)
