import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'close': [100, 105, 110],
    })


def test_buy_and_hold_basic_fields(backtester, price_df):
    result = backtester.buy_and_hold(price_df)

    assert result['buy_date'] == '2024-01-01'
    assert result['sell_date'] == '2024-01-03'
    assert result['buy_price'] == 100
    assert result['sell_price'] == 110
    assert result['holding_days'] == 3
    assert result['include_cost'] is True


def test_buy_and_hold_returns_with_cost(backtester, price_df):
    # buy 100 -> cost 100*(1+0.001425)=100.1425
    # sell 110 -> income 110*(1-0.001425-0.003)=109.615
    # returns = (109.615-100.1425)/100.1425*100 = 9.36
    result = backtester.buy_and_hold(price_df, include_cost=True)

    assert result['returns'] == pytest.approx(9.36, abs=0.01)


def test_buy_and_hold_returns_without_cost(backtester, price_df):
    # (110-100)/100*100 = 10.0
    result = backtester.buy_and_hold(price_df, include_cost=False)

    assert result['returns'] == 10.0
    assert result['include_cost'] is False


def test_buy_and_hold_returns_negative_with_cost(backtester):
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'close': [100, 90],
    })

    result = backtester.buy_and_hold(df, include_cost=True)

    assert result['returns'] == -10.53


def test_buy_and_hold_sorts_by_date(backtester):
    # 輸入未依日期排序，函式應先排序再取第一筆/最後一筆
    df = pd.DataFrame({
        'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
        'close': [110, 100, 105],
    })

    result = backtester.buy_and_hold(df, include_cost=False)

    assert result['buy_date'] == '2024-01-01'
    assert result['buy_price'] == 100
    assert result['sell_date'] == '2024-01-03'
    assert result['sell_price'] == 110


def test_buy_and_hold_equity_curve(backtester, price_df):
    result = backtester.buy_and_hold(price_df, include_cost=True)
    equity_curve = result['equity_curve']

    # shares = 1,000,000 / (100 * 1.001425)
    shares = 1_000_000 / (100 * 1.001425)
    assert list(equity_curve.columns) == ['date', 'close']
    assert len(equity_curve) == len(price_df)
    assert equity_curve.iloc[0]['close'] == pytest.approx(shares * 100)
    assert equity_curve.iloc[-1]['close'] == pytest.approx(
        shares * 110 * (1 - Backtester.COMMISSION_RATE - Backtester.TAX_RATE)
    )


def test_buy_and_hold_single_day(backtester):
    # 買入賣出同一天，報酬應為負（手續費+稅）
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'close': [100],
    })

    result = backtester.buy_and_hold(df, include_cost=True)

    assert result['buy_price'] == result['sell_price'] == 100
    assert result['holding_days'] == 1
    assert result['returns'] < 0
