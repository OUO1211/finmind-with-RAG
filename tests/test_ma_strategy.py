import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    # short_window=2, long_window=3；open 特意跟 close 設不同值，用來驗證
    # 策略真的是用「隔天開盤價」成交，不是誤用收盤價
    # idx0,1 被 dropna 排除（long_ma 需要 3 筆資料才有值）
    # idx2 (01-03): short_ma=10.00, long_ma=10.00        -> 持平
    # idx3 (01-04): short_ma=15.00, long_ma=13.33 (>)    -> 黃金交叉訊號，隔天(01-05)開盤買進
    # idx4 (01-05): short_ma=20.00, long_ma=16.67 (>)    -> 持有中
    # idx5 (01-06): short_ma=20.00, long_ma=20.00        -> 持平
    # idx6 (01-07): short_ma=12.50, long_ma=15.00 (<)    -> 死亡交叉訊號，隔天(01-08)開盤賣出
    # idx7 (01-08): short_ma=5.00,  long_ma=10.00 (<)    -> 已空手
    # idx8,9: short_ma=long_ma=5.00                      -> 持平
    return pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 11)],
        'open': [9, 9, 9, 18, 19, 19, 6, 5, 5, 5],
        'close': [10, 10, 10, 20, 20, 20, 5, 5, 5, 5],
    })


def test_ma_strategy_buy_and_sell(backtester, price_df):
    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    # 01-04 產生黃金交叉訊號，隔天 01-05 開盤(19)才成交
    assert trade['buy_date'] == '2024-01-05'
    assert trade['buy_price'] == 19
    # 01-07 產生死亡交叉訊號，隔天 01-08 開盤(5)才成交
    assert trade['sell_date'] == '2024-01-08'
    assert trade['sell_price'] == 5


def test_ma_strategy_returns_with_cost(backtester, price_df):
    # buy 19 -> cost 19*(1+0.001425)=19.027075
    # sell 5 -> income 5*(1-0.001425-0.003)=4.977875
    # return = (4.977875-19.027075)/19.027075*100 = -73.84
    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)

    assert result['trades'][0]['return'] == -73.84
    assert result['total_return'] == -73.84
    assert result['include_cost'] is True


def test_ma_strategy_returns_without_cost(backtester, price_df):
    # (5-19)/19*100 = -73.68
    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=False)

    assert result['trades'][0]['return'] == -73.68
    assert result['total_return'] == -73.68
    assert result['include_cost'] is False


def test_ma_strategy_no_trades_when_no_crossover(backtester):
    # 均線始終持平，短均線從未大於長均線，不應觸發買進
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'open': [10, 10, 10, 10],
        'close': [10, 10, 10, 10],
    })

    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_ma_strategy_still_holding_produces_no_trade(backtester):
    # 只出現黃金交叉未出現死亡交叉，不應產生已完成交易紀錄
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 6)],
        'open': [9, 9, 9, 18, 24],
        'close': [10, 10, 10, 20, 25],
    })

    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []


def test_ma_strategy_no_execution_on_last_day_signal(backtester):
    # 訊號出現在資料的最後一天時，沒有「隔天」可以成交，不應產生任何交易
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 5)],
        'open': [9, 9, 9, 18],
        'close': [10, 10, 10, 20],
    })

    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []


def test_ma_strategy_sorts_by_date(backtester, price_df):
    # 輸入未依日期排序，函式應先排序再依序計算均線與判斷買賣
    shuffled_df = price_df.sample(frac=1, random_state=42).reset_index(drop=True)

    result = backtester.ma_strategy(shuffled_df, short_window=2, long_window=3, include_cost=True)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-05'
    assert result['trades'][0]['sell_date'] == '2024-01-08'


def test_ma_strategy_equity_curve(backtester, price_df):
    result = backtester.ma_strategy(price_df, short_window=2, long_window=3, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    # dropna 排除前兩筆（idx0, idx1），僅剩 8 筆資料
    assert len(equity_curve) == len(price_df) - 2
    # 01-03、01-04 都還沒買進（訊號要到 01-05 開盤才執行），維持初始資金
    assert equity_curve.iloc[0]['close'] == 1_000_000
    assert equity_curve.iloc[1]['close'] == 1_000_000
    # 01-05 以開盤價 19 買進，equity 用當天收盤價 20 估市值
    shares = 1_000_000 / 19
    assert equity_curve.iloc[2]['close'] == pytest.approx(shares * 20)
    assert equity_curve.iloc[3]['close'] == pytest.approx(shares * 20)
