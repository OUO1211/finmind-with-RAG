import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'close': [100, 100, 105, 110, 90],
    })


@pytest.fixture
def per_df():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'PER': [20, 10, 12, 30, 18],
    })


def test_pe_strategy_buy_and_sell(backtester, price_df, per_df):
    # PER 10 < buy_pe(15) 於 01-02 買進，PER 30 > sell_pe(25) 於 01-04 賣出
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    assert trade['buy_date'] == '2024-01-02'
    assert trade['buy_price'] == 100
    assert trade['sell_date'] == '2024-01-04'
    assert trade['sell_price'] == 110


def test_pe_strategy_returns_with_cost(backtester, price_df, per_df):
    # buy 100 -> cost 100*(1+0.001425)=100.1425
    # sell 110 -> income 110*(1-0.001425-0.003)=109.51325
    # return = (109.51325-100.1425)/100.1425*100 = 9.36
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['trades'][0]['return'] == 9.36
    assert result['total_return'] == 9.36
    assert result['include_cost'] is True


def test_pe_strategy_returns_without_cost(backtester, price_df, per_df):
    # (110-100)/100*100 = 10.0
    result = backtester.pe_strategy(price_df, per_df, include_cost=False)

    assert result['trades'][0]['return'] == 10.0
    assert result['total_return'] == 10.0
    assert result['include_cost'] is False


def test_pe_strategy_custom_thresholds(backtester, price_df, per_df):
    # buy_pe=25：01-01 PER=20 < 25 即買進；sell_pe=15：01-03 PER=12 < 15 尚未賣出，需 PER > 15
    # 01-02 PER=10 已持有中不會再買；01-03 PER=12 未 >15 不賣；01-04 PER=30 >15 賣出
    result = backtester.pe_strategy(price_df, per_df, buy_pe=25, sell_pe=15, include_cost=False)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-01'
    assert result['trades'][0]['sell_date'] == '2024-01-04'


def test_pe_strategy_no_trades_when_pe_never_below_buy_pe(backtester, price_df, per_df):
    result = backtester.pe_strategy(price_df, per_df, buy_pe=5, sell_pe=25, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_pe_strategy_still_holding_produces_no_trade(backtester):
    # 只買未賣（PER 始終未超過 sell_pe），不應產生任何完成的交易紀錄
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'close': [100, 105],
    })
    per_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'PER': [10, 12],
    })

    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []


def test_pe_strategy_merge_keeps_only_common_dates(backtester):
    # per_df 缺少 01-03，該日應被 inner join 排除
    price_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
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
    price_df = pd.DataFrame({
        'date': ['2024-01-02', '2024-01-01'],
        'close': [100, 100],
    })
    per_df = pd.DataFrame({
        'date': ['2024-01-02', '2024-01-01'],
        'PER': [30, 10],
    })

    result = backtester.pe_strategy(price_df, per_df, include_cost=True)

    # 排序後：01-01 PER=10 買進，01-02 PER=30 賣出
    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-01'
    assert result['trades'][0]['sell_date'] == '2024-01-02'


def test_pe_strategy_equity_curve(backtester, price_df, per_df):
    result = backtester.pe_strategy(price_df, per_df, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    assert len(equity_curve) == len(price_df)
    # 01-01 尚未買進，portfolio 維持初始資金
    assert equity_curve.iloc[0]['close'] == 1_000_000
    # 01-02 以收盤價 100 買進
    shares = 1_000_000 / 100
    assert equity_curve.iloc[1]['close'] == pytest.approx(shares * 100)
    assert equity_curve.iloc[2]['close'] == pytest.approx(shares * 105)
