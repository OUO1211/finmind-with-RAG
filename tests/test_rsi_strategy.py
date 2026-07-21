import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    # period=3, buy_rsi=30, sell_rsi=70
    # RSI（dropna 後）: idx0=0.00, idx1=33.33, idx2=66.67, idx3=100.00, idx4=100.00
    # idx0: RSI=0.00   (<30) -> 於 close=85 買進
    # idx1: RSI=33.33  持有中
    # idx2: RSI=66.67  持有中
    # idx3: RSI=100.00 (>70) -> 於 close=100 賣出
    # idx4: RSI=100.00 已空手，不再買進
    return pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 9)],
        'close': [100, 95, 90, 85, 90, 95, 100, 105],
    })


def test_rsi_strategy_buy_and_sell(backtester, price_df):
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    assert trade['buy_date'] == '2024-01-04'
    assert trade['buy_price'] == 85
    assert trade['sell_date'] == '2024-01-07'
    assert trade['sell_price'] == 100


def test_rsi_strategy_returns_with_cost(backtester, price_df):
    # buy 85 -> cost 85*(1+0.001425)=85.121125
    # sell 100 -> income 100*(1-0.001425-0.003)=99.5575
    # return = (99.5575-85.121125)/85.121125*100 = 16.96
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['trades'][0]['return'] == 16.96
    assert result['total_return'] == 16.96
    assert result['include_cost'] is True


def test_rsi_strategy_returns_without_cost(backtester, price_df):
    # (100-85)/85*100 = 17.65
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=False)

    assert result['trades'][0]['return'] == 17.65
    assert result['total_return'] == 17.65
    assert result['include_cost'] is False


def test_rsi_strategy_no_trades_when_rsi_stays_neutral(backtester):
    # 股價持平，漲跌幅為 0，RSI 無法計算出低於 buy_rsi 的訊號
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 6)],
        'close': [100, 100, 100, 100, 100],
    })

    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_rsi_strategy_still_holding_produces_no_trade(backtester):
    # 只出現超賣買進訊號，RSI 未曾超過 sell_rsi，不應產生已完成交易紀錄
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 6)],
        'close': [100, 95, 90, 85, 87],
    })

    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []


def test_rsi_strategy_sorts_by_date(backtester, price_df):
    # 輸入未依日期排序，函式應先排序再依序計算 RSI 與判斷買賣
    shuffled_df = price_df.sample(frac=1, random_state=42).reset_index(drop=True)

    result = backtester.rsi_strategy(shuffled_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-04'
    assert result['trades'][0]['sell_date'] == '2024-01-07'


def test_rsi_strategy_equity_curve(backtester, price_df):
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    # dropna 排除前 3 筆（period=3 需 rolling 資料才有值），僅剩 5 筆資料
    assert len(equity_curve) == len(price_df) - 3
    # 第一筆（01-04）以收盤價 85 買進
    shares = 1_000_000 / 85
    assert equity_curve.iloc[0]['close'] == pytest.approx(shares * 85)
    # 01-07 賣出後，資金維持在賣出當下的市值
    assert equity_curve.iloc[3]['close'] == pytest.approx(shares * 100)
    assert equity_curve.iloc[4]['close'] == pytest.approx(shares * 100)
