import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    # open 特意跟 close 設不同值，用來驗證策略真的是用「隔天開盤價」成交
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'open': [99, 99, 104, 109, 89],
        'close': [100, 100, 105, 110, 90],
    })


@pytest.fixture
def per_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'PER': [20, 10, 12, 30, 18],
    })


def test_pe_strategy_buy_and_sell(backtester, price_df, per_df):
    # PER 10 < buy_pe(15) 於 01-02 產生買進訊號，隔天 01-03 開盤(104)才成交
    # PER 30 > sell_pe(25) 於 01-04 產生賣出訊號，隔天 01-05 開盤(89)才成交
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    assert trade['buy_date'] == '2024-01-03'
    assert trade['buy_price'] == 104
    assert trade['sell_date'] == '2024-01-05'
    assert trade['sell_price'] == 89


def test_pe_strategy_returns_with_cost(backtester, price_df, per_df):
    # buy 104 -> cost 104*(1+0.001425)=104.1482
    # sell 89 -> income 89*(1-0.001425-0.003)=88.60958
    # return = (88.60958-104.1482)/104.1482*100 = -14.92
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['trades'][0]['return'] == -14.92
    assert result['total_return'] == -14.92
    assert result['include_cost'] is True


def test_pe_strategy_returns_without_cost(backtester, price_df, per_df):
    # (89-104)/104*100 = -14.42
    result = backtester.pe_strategy(price_df, per_df, include_cost=False)

    assert result['trades'][0]['return'] == -14.42
    assert result['total_return'] == -14.42
    assert result['include_cost'] is False


def test_pe_strategy_custom_thresholds(backtester, price_df, per_df):
    # buy_pe=25：01-01 PER=20<25 產生訊號，但 01-01 是第一天沒有「前一天」可執行，
    # 訊號作廢；01-02 PER=10<25 產生訊號，隔天 01-03 開盤(104)... 但實際上一開始持有
    # 狀態要看哪個訊號最先「可執行」：01-02 這天執行的是 01-01 的訊號（True），
    # 所以買進發生在 01-02 開盤價(99)
    # sell_pe=15：01-04 PER=30>15 產生訊號，隔天 01-05 開盤(89)賣出
    result = backtester.pe_strategy(price_df, per_df, buy_pe=25, sell_pe=15, include_cost=False)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-02'
    assert result['trades'][0]['buy_price'] == 99
    assert result['trades'][0]['sell_date'] == '2024-01-05'
    assert result['trades'][0]['sell_price'] == 89


def test_pe_strategy_no_trades_when_pe_never_below_buy_pe(backtester, price_df, per_df):
    result = backtester.pe_strategy(price_df, per_df, buy_pe=5, sell_pe=25, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_pe_strategy_still_holding_produces_no_trade(backtester):
    # 01-01 PER=10 產生訊號但當天無法執行（沒有前一天），01-02 執行買進，
    # 01-03 PER 仍未超過 sell_pe，不應產生已完成的交易紀錄
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [99, 99, 104],
        'close': [100, 100, 105],
    })
    per_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'PER': [10, 12, 12],
    })

    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []
    # 用 equity_curve 確認買進真的執行了（01-02 開盤價 99 買進），
    # 不是「訊號跟執行都沒發生」導致的假陽性
    shares = 1_000_000 / 99
    assert result['equity_curve'].iloc[1]['close'] == pytest.approx(shares * 100)


def test_pe_strategy_merge_keeps_only_common_dates(backtester):
    # per_df 缺少 01-03，該日應被 inner join 排除
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [99, 99, 104],
        'close': [100, 105, 110],
    })
    per_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'PER': [20, 10],
    })

    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert len(result['equity_curve']) == 2
    assert list(result['equity_curve']['date']) == ['2024-01-01', '2024-01-02']


def test_pe_strategy_sorts_by_date(backtester):
    # 輸入未依日期排序，函式應先排序再依序判斷買賣
    # 排序後：01-01 PER=10 產生訊號，隔天 01-02 開盤(99)執行買進
    price_df = pd.DataFrame({
        'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
        'open': [104, 99, 99],
        'close': [105, 100, 100],
    })
    per_df = pd.DataFrame({
        'date': ['2024-01-03', '2024-01-01', '2024-01-02'],
        'PER': [30, 10, 12],
    })

    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    shares = 1_000_000 / 99
    equity_curve = result['equity_curve']
    assert list(equity_curve['date']) == ['2024-01-01', '2024-01-02', '2024-01-03']
    assert equity_curve.iloc[0]['close'] == 1_000_000
    assert equity_curve.iloc[1]['close'] == pytest.approx(shares * 100)


def test_pe_strategy_equity_curve(backtester, price_df, per_df):
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    assert len(equity_curve) == len(price_df)
    # 01-01、01-02 尚未買進（訊號要到 01-03 開盤才執行），維持初始資金
    assert equity_curve.iloc[0]['close'] == 1_000_000
    assert equity_curve.iloc[1]['close'] == 1_000_000
    # 01-03 以開盤價 104 買進，equity 用當天收盤價 105 估市值
    shares = 1_000_000 / 104
    assert equity_curve.iloc[2]['close'] == pytest.approx(shares * 105)
