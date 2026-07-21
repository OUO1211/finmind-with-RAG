import pandas as pd
import pytest
from src.risk_manager import RiskManager


class FakeDataService:
    """取代真實 DataService，避免 heat_check 測試觸發快取/外部 API"""

    def __init__(self, df):
        self.df = df

    def get_data(self, stock_id, data_type, start_date, end_date):
        return self.df


@pytest.fixture
def risk_manager():
    # heat_check 只用到 self.data_service，跳過 __init__ 以避免建立真實 DataService/Backtester
    return RiskManager.__new__(RiskManager)


def make_price_df(closes):
    # RSI(14) 與 MA60 皆需要至少 60 筆資料才有值，用序號產生足量日期
    dates = [f'2024-{(1 + i // 28):02d}-{(1 + i % 28):02d}' for i in range(len(closes))]
    return pd.DataFrame({'date': dates, 'close': closes})


def test_heat_check_no_signals_when_normal(risk_manager):
    # 前 45 天平盤 100，最後 15 天小幅震盪收在 102：RSI 中性、股價略高於 MA60、無乖離/回撤
    closes = [100] * 45 + [101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 102]
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert result['rsi'] == 53.33
    assert result['above_ma'] == True
    assert result['signals'] == []


def test_heat_check_overbought_and_high_bias(risk_manager):
    # 前 46 天平盤 100，最後 14 天急漲至 140：RSI 超買 + 乖離率過高 (>30%)
    closes = [100] * 46 + [round(100 + i * 40 / 14, 2) for i in range(1, 15)]
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert result['rsi'] == 100.0
    assert result['bias'] == 33.33
    assert 'RSI 超買' in result['signals']
    assert any('乖離率過高' in s for s in result['signals'])


def test_heat_check_oversold_below_ma_and_bear_drawdown(risk_manager):
    # 前 46 天平盤 100，最後 14 天急跌至 65：RSI 超賣 + 跌破 MA60 + 回撤 > 20%（空頭）
    closes = [100] * 46 + [round(100 - i * 35 / 14, 2) for i in range(1, 15)]
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert result['rsi'] == 0.0
    assert result['above_ma'] == False
    assert result['drawdown'] == -35.0
    assert result['signals'] == ['RSI 超賣', '跌破 MA60', '空頭（回撤 > 20%）']


def test_heat_check_correction_drawdown(risk_manager):
    # 回撤介於 -10% ~ -20% 之間 -> 觸發「修正」訊號（非空頭、非小回檔）
    closes = [100] * 46 + [round(100 - i * 15 / 14, 2) for i in range(1, 15)]
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert result['drawdown'] == -15.0
    assert any('修正' in s for s in result['signals'])


def test_heat_check_small_pullback_drawdown(risk_manager):
    # 回撤介於 -5% ~ -10% 之間 -> 觸發「小回檔」訊號
    closes = [100] * 46 + [round(100 - i * 8 / 14, 2) for i in range(1, 15)]
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert result['drawdown'] == -8.0
    assert '小回檔' in result['signals']


def test_heat_check_return_keys(risk_manager):
    closes = [100] * 60
    price_df = make_price_df(closes)
    risk_manager.data_service = FakeDataService(price_df)

    result = risk_manager.heat_check('2330', price_df['date'].iloc[0], price_df['date'].iloc[-1])

    assert set(result.keys()) == {'rsi', 'price', 'ma60', 'above_ma', 'drawdown', 'bias', 'signals'}
