import pandas as pd
import pytest
from src.risk_manager import RiskManager


class FakeDataService:
    """取代真實 DataService，避免 get_rsi 測試觸發快取/外部 API"""

    def __init__(self, df):
        self.df = df
        self.calls = []

    def get_data(self, stock_id, data_type, start_date, end_date):
        self.calls.append((stock_id, data_type, start_date, end_date))
        return self.df


@pytest.fixture
def risk_manager():
    # get_rsi 只用到 self.data_service，跳過 __init__ 以避免建立真實 DataService/Backtester
    return RiskManager.__new__(RiskManager)


def test_get_rsi_oversold(risk_manager):
    # 連續下跌 -> avg_loss 恆為正、avg_gain 恆為 0 -> RSI = 0（超賣）
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 7)],
        'close': [100, 95, 90, 85, 80, 75],
    })
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.get_rsi('2330', '2024-01-01', '2024-01-06', period=3)

    assert result == 0.0


def test_get_rsi_overbought(risk_manager):
    # 連續上漲 -> avg_loss 恆為 0 -> RSI = 100（超買）
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 7)],
        'close': [100, 105, 110, 115, 120, 125],
    })
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.get_rsi('2330', '2024-01-01', '2024-01-06', period=3)

    assert result == 100.0


def test_get_rsi_neutral(risk_manager):
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 7)],
        'close': [100, 105, 98, 103, 99, 101],
    })
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.get_rsi('2330', '2024-01-01', '2024-01-06', period=3)

    assert result == 63.64


def test_get_rsi_passes_correct_args_to_data_service(risk_manager):
    price_df = pd.DataFrame({
        'date': [f'2024-01-{d:02d}' for d in range(1, 7)],
        'close': [100, 105, 98, 103, 99, 101],
    })
    fake_service = FakeDataService(price_df)
    risk_manager.data_service = fake_service

    risk_manager.get_rsi('2330', '2024-01-01', '2024-01-06', period=3)

    assert fake_service.calls == [('2330', 'stock_price', '2024-01-01', '2024-01-06')]


def test_get_rsi_sorts_by_date(risk_manager):
    # 傳入未依日期排序的資料，get_rsi 應先排序再計算，結果需與已排序資料一致
    shuffled_df = pd.DataFrame({
        'date': ['2024-01-04', '2024-01-01', '2024-01-06', '2024-01-02', '2024-01-05', '2024-01-03'],
        'close': [85, 100, 75, 95, 80, 90],
    })
    risk_manager.data_service = FakeDataService(shuffled_df)

    result = risk_manager.get_rsi('2330', '2024-01-01', '2024-01-06', period=3)

    assert result == 0.0
