import pandas as pd
import pytest
from src.backtester import Backtester


@pytest.fixture
def backtester():
    return Backtester()


@pytest.fixture
def price_df():
    # period=3；min/max 皆為 close ± 2，只用來驅動 RSV/K/D 的漲跌趨勢；
    # open 特意跟 close 錯開 1，用來驗證策略真的是用「隔天開盤價」成交
    # K/D（dropna 後，period=3）:
    # idx0 (01-03): K=38.10, D=46.03
    # idx1 (01-04): K=30.16, D=40.74
    # idx2 (01-05): K=24.87, D=35.45  (prev_k < prev_d)
    # idx3 (01-06): K=42.50, D=37.80  (curr_k > curr_d) -> 黃金交叉訊號，
    #                                    隔天(01-07)開盤(89)才成交
    # idx4 (01-07): K=56.91, D=44.17  持有中
    # idx5 (01-08): K=66.51, D=51.62  持有中
    # idx6 (01-09): K=72.91, D=58.71  持有中
    # idx7 (01-10): K=77.18, D=64.87  (prev_k > prev_d)
    # idx8 (01-11): K=58.86, D=62.87  (curr_k < curr_d) -> 死亡交叉訊號，
    #                                    隔天(01-12)開盤(94)才成交
    # idx9~11: 空手
    closes = [100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 100, 95, 90, 85]
    return pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, len(closes) + 1)],
        'open': [c - 1 for c in closes],
        'close': closes,
        'min': [c - 2 for c in closes],
        'max': [c + 2 for c in closes],
    })


def test_kd_strategy_buy_and_sell(backtester, price_df):
    result = backtester.kd_strategy(price_df, period=3, include_cost=True)

    assert result['total_trades'] == 1
    trade = result['trades'][0]
    assert trade['buy_date'] == '2024-01-07'
    assert trade['buy_price'] == 89
    assert trade['sell_date'] == '2024-01-12'
    assert trade['sell_price'] == 94


def test_kd_strategy_returns_with_cost(backtester, price_df):
    # buy 89 -> cost 89*(1+0.001425)=89.12683
    # sell 94 -> income 94*(1-0.001425-0.003)=93.5809
    # return = (93.5809-89.12683)/89.12683*100 = 5.0
    result = backtester.kd_strategy(price_df, period=3, include_cost=True)

    assert result['trades'][0]['return'] == 5.0
    assert result['total_return'] == 5.0
    assert result['include_cost'] is True


def test_kd_strategy_returns_without_cost(backtester, price_df):
    # (94-89)/89*100 = 5.62
    result = backtester.kd_strategy(price_df, period=3, include_cost=False)

    assert result['trades'][0]['return'] == 5.62
    assert result['total_return'] == 5.62
    assert result['include_cost'] is False


def test_kd_strategy_no_trades_when_flat(backtester):
    # 股價持平，RSV/K/D 皆不變動，K 與 D 不會交叉
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 8)],
        'open': [100] * 7,
        'close': [100] * 7,
        'min': [98] * 7,
        'max': [102] * 7,
    })

    result = backtester.kd_strategy(price_df, period=3, include_cost=True)

    assert result['total_trades'] == 0
    assert result['total_return'] == 0
    assert result['trades'] == []


def test_kd_strategy_still_holding_produces_no_trade(backtester, price_df):
    # 只保留到黃金交叉訊號執行（01-07）之後、死亡交叉發生前的資料
    still_holding_df = price_df.iloc[:8].reset_index(drop=True)

    result = backtester.kd_strategy(still_holding_df, period=3, include_cost=True)

    assert result['total_trades'] == 0
    assert result['trades'] == []
    # 用 equity_curve 確認買進真的執行了（01-07 開盤價 89 買進）
    shares = 1_000_000 / (89 * (1 + Backtester.COMMISSION_RATE))
    assert result['equity_curve'].iloc[4]['close'] == pytest.approx(shares * 90)


def test_kd_strategy_sorts_by_date(backtester, price_df):
    # 輸入未依日期排序，函式應先排序再依序計算 K/D 與判斷買賣
    shuffled_df = price_df.sample(frac=1, random_state=42).reset_index(drop=True)

    result = backtester.kd_strategy(shuffled_df, period=3, include_cost=True)

    assert result['total_trades'] == 1
    assert result['trades'][0]['buy_date'] == '2024-01-07'
    assert result['trades'][0]['sell_date'] == '2024-01-12'


def test_kd_strategy_equity_curve(backtester, price_df):
    result = backtester.kd_strategy(price_df, period=3, include_cost=True)
    equity_curve = result['equity_curve']

    assert list(equity_curve.columns) == ['date', 'close']
    # dropna 排除前 2 筆（period=3 需 rolling 資料才有值），僅剩 12 筆資料
    assert len(equity_curve) == len(price_df) - 2
    # 買進前（01-03 ~ 01-06）維持初始資金（訊號要到 01-07 開盤才執行）
    assert equity_curve.iloc[0]['close'] == 1_000_000
    assert equity_curve.iloc[3]['close'] == 1_000_000
    # 01-07 以開盤價 89 買進，equity 用當天收盤價 90 估市值
    shares = 1_000_000 / (89 * (1 + Backtester.COMMISSION_RATE))
    assert equity_curve.iloc[4]['close'] == pytest.approx(shares * 90)
    # 01-12 賣出後（開盤價 94），資金維持在賣出當下的市值
    sale_value = shares * 94 * (1 - Backtester.COMMISSION_RATE - Backtester.TAX_RATE)
    assert equity_curve.iloc[9]['close'] == pytest.approx(sale_value)
    assert equity_curve.iloc[10]['close'] == pytest.approx(sale_value)
