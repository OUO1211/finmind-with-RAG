import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    # period=3, buy_rsi=30, sell_rsi=70；open 特意跟 close 設不同值，
    # 用來驗證策略真的是用「隔天開盤價」成交，不是誤用收盤價
    # RSI（dropna 後，9 天資料，前 3 天因 rolling(3) 缺值被排除）：
    # 01-04: RSI=0.00   -> 產生買進訊號，但當天是第一筆，沒有前一天可執行，訊號作廢
    # 01-05: RSI=0.00   -> 也 <30，隔天(01-06)... 不過因為 01-04 訊號已經
    #                      ready，實際上 01-05 執行的是「01-04 的訊號」，於開盤價 84 買進
    # 01-06: RSI=33.33  持有中
    # 01-07: RSI=66.67  持有中
    # 01-08: RSI=100.00 -> 產生賣出訊號，隔天 01-09 開盤(104)賣出
    # 01-09: RSI=100.00 已空手，不再買進
    return pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 10)],
        'open': [99, 99, 94, 89, 84, 89, 94, 99, 104],
        'close': [100, 100, 95, 90, 85, 90, 95, 100, 105],
    })


def test_rsi_strategy_buy_and_sell(backtester, price_df):
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    assert trade['buy_date'] == '2024-01-05'
    assert trade['buy_price'] == 84
    assert trade['sell_date'] == '2024-01-09'
    assert trade['sell_price'] == 104


def test_rsi_strategy_returns_with_cost(backtester, price_df):
    # buy 84 -> cost 84*(1+0.001425)=84.11970
    # sell 104 -> income 104*(1-0.001425-0.003)=103.5382
    # return = (103.5382-84.1197)/84.1197*100 = 23.09
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['trades'][0]['return'] == 23.09
    assert result['total_return'] == 23.09
    assert result['include_cost'] is True


def test_rsi_strategy_returns_without_cost(backtester, price_df):
    # (104-84)/84*100 = 23.81
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=False)

    assert result['trades'][0]['return'] == 23.81
    assert result['total_return'] == 23.81
    assert result['include_cost'] is False


def test_rsi_strategy_no_trades_when_rsi_stays_neutral(backtester):
    # 股價持平，漲跌幅為 0，RSI 無法計算出低於 buy_rsi 的訊號
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 6)],
        'open': [100, 100, 100, 100, 100],
        'close': [100, 100, 100, 100, 100],
    })

    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_rsi_strategy_still_holding_produces_no_trade(backtester):
    # 只出現超賣買進訊號，RSI 未曾超過 sell_rsi，不應產生已完成交易紀錄
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 7)],
        'open': [99, 94, 89, 84, 79, 82],
        'close': [100, 95, 90, 85, 80, 82],
    })

    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []
    # 用 equity_curve 確認買進真的執行了（01-05 開盤價 79 買進），
    # 不是「訊號跟執行都沒發生」導致的假陽性
    shares = 1_000_000 / 79
    assert result['equity_curve'].iloc[1]['close'] == pytest.approx(shares * 80)


def test_rsi_strategy_sorts_by_date(backtester, price_df):
    # 輸入未依日期排序，函式應先排序再依序計算 RSI 與判斷買賣
    shuffled_df = price_df.sample(frac=1, random_state=42).reset_index(drop=True)

    result = backtester.rsi_strategy(shuffled_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-05'
    assert result['trades'][0]['sell_date'] == '2024-01-09'


def test_rsi_strategy_equity_curve(backtester, price_df):
    result = backtester.rsi_strategy(price_df, period=3, buy_rsi=30, sell_rsi=70, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    # dropna 排除前 3 筆（period=3 需 rolling 資料才有值），僅剩 6 筆資料
    assert len(equity_curve) == len(price_df) - 3
    # 01-04 尚未買進（訊號要到 01-05 開盤才執行），維持初始資金
    assert equity_curve.iloc[0]['close'] == 1_000_000
    # 01-05 以開盤價 84 買進，equity 用當天收盤價 85 估市值
    shares = 1_000_000 / 84
    assert equity_curve.iloc[1]['close'] == pytest.approx(shares * 85)
    # 01-09 賣出後，資金維持在賣出當下的市值
    assert equity_curve.iloc[5]['close'] == pytest.approx(shares * 104)
