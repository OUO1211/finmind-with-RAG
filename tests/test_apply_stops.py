import pandas as pd
import pytest
from src.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    # apply_stops 不需要 DataService / Backtester，跳過 __init__ 以避免外部依賴
    return RiskManager.__new__(RiskManager)


@pytest.fixture
def price_df():
    # 買進 01-01（close=100）之後的走勢：01-02=105, 01-03=112(+12%), 01-04=90(-10%), 01-05=95, 01-06=100, 01-07=105
    return pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 8)],
        'close': [100, 105, 112, 90, 95, 100, 105],
    })


def test_apply_stops_take_profit_triggers_early_exit(risk_manager, price_df):
    original_result = {
        'trades': [
            {'buy_date': '2024-01-01', 'sell_date': '2024-01-07', 'buy_price': 100, 'sell_price': 105, 'return': 5.0},
        ]
    }

    result = risk_manager.apply_stops(price_df, original_result, stop_loss=-10, take_profit=10)

    trade = result['trades'][0]
    # 01-03 漲幅 12% > take_profit(10%) -> 提前於 01-03 出場
    assert trade['sell_date'] == '2024-01-03'
    assert trade['sell_price'] == 112
    assert trade['return'] == 12.0
    assert result['total_trades'] == 1
    assert result['total_return'] == 12.0


def test_apply_stops_stop_loss_triggers_early_exit(risk_manager, price_df):
    original_result = {
        'trades': [
            {'buy_date': '2024-01-01', 'sell_date': '2024-01-07', 'buy_price': 100, 'sell_price': 105, 'return': 5.0},
        ]
    }

    result = risk_manager.apply_stops(price_df, original_result, stop_loss=-8, take_profit=20)

    trade = result['trades'][0]
    # 01-04 跌幅 -10% < stop_loss(-8%) -> 提前於 01-04 出場
    assert trade['sell_date'] == '2024-01-04'
    assert trade['sell_price'] == 90
    assert trade['return'] == -10.0
    assert result['total_return'] == -10.0


def test_apply_stops_no_trigger_keeps_original_trade(risk_manager, price_df):
    original_result = {
        'trades': [
            {'buy_date': '2024-01-01', 'sell_date': '2024-01-07', 'buy_price': 100, 'sell_price': 105, 'return': 5.0},
        ]
    }

    # 停損停利門檻極寬鬆，區間內漲跌幅都不會觸發
    result = risk_manager.apply_stops(price_df, original_result, stop_loss=-50, take_profit=50)

    assert result['trades'][0] == original_result['trades'][0]
    assert result['total_trades'] == 1
    assert result['total_return'] == 5.0


def test_apply_stops_multiple_trades(risk_manager, price_df):
    original_result = {
        'trades': [
            {'buy_date': '2024-01-01', 'sell_date': '2024-01-03', 'buy_price': 100, 'sell_price': 112, 'return': 12.0},
            {'buy_date': '2024-01-04', 'sell_date': '2024-01-07', 'buy_price': 90, 'sell_price': 105, 'return': 16.67},
        ]
    }

    result = risk_manager.apply_stops(price_df, original_result, stop_loss=-10, take_profit=10)

    assert result['total_trades'] == 2
    # 第一筆：01-01買進，01-02漲幅5%未觸發，01-03漲幅12%觸發停利
    assert result['trades'][0]['sell_date'] == '2024-01-03'
    assert result['trades'][0]['return'] == 12.0
    # 第二筆：01-04買進(90)，01-06漲至100(+11.11%)觸發停利，比原本01-07更早出場
    assert result['trades'][1]['sell_date'] == '2024-01-06'
    assert result['trades'][1]['sell_price'] == 100
    assert result['trades'][1]['return'] == 11.11
    assert result['total_return'] == 23.11


def test_apply_stops_no_trades(risk_manager, price_df):
    original_result = {'trades': []}

    result = risk_manager.apply_stops(price_df, original_result, stop_loss=-10, take_profit=10)

    assert result['trades'] == []
    assert result['total_trades'] == 0
    assert result['total_return'] == 0
