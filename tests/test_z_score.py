import pandas as pd
import pytest
from src.risk_manager import RiskManager


class FakeDataService:
    """依 data_type 回傳對應假資料，取代真實 DataService 呼叫外部 API"""

    def __init__(self, data_by_type):
        self.data_by_type = data_by_type
        self.calls = []

    def get_data(self, stock_id, data_type, start_date, end_date):
        self.calls.append((stock_id, data_type, start_date, end_date))
        return self.data_by_type[data_type]


@pytest.fixture
def risk_manager():
    # z_score 只用到 self.data_service，跳過 __init__ 以避免建立真實 DataService/Backtester
    return RiskManager.__new__(RiskManager)


def make_balance_sheet(date, current_assets, current_liabilities, total_assets,
                        equity, retained_earnings=None):
    rows = [
        {'date': date, 'type': 'CurrentAssets', 'value': current_assets},
        {'date': date, 'type': 'CurrentLiabilities', 'value': current_liabilities},
        {'date': date, 'type': 'TotalAssets', 'value': total_assets},
        {'date': date, 'type': 'EquityAttributableToOwnersOfParent', 'value': equity},
    ]
    if retained_earnings is not None:
        rows.append({'date': date, 'type': 'RetainedEarnings', 'value': retained_earnings})
    return pd.DataFrame(rows)


def make_financial_statement(date, operating_income, revenue):
    return pd.DataFrame([
        {'date': date, 'type': 'OperatingIncome', 'value': operating_income},
        {'date': date, 'type': 'Revenue', 'value': revenue},
    ])


def test_z_score_safe_zone(risk_manager):
    # A=0.3, B=0.6, C=0.3, E=1.0 -> z=1.2*0.3+1.4*0.6+3.3*0.3+1.0*1.0=3.19 (>2.99 安全)
    bs = make_balance_sheet('2024-Q1', current_assets=500, current_liabilities=200,
                             total_assets=1000, equity=700, retained_earnings=600)
    fs = make_financial_statement('2024-Q1', operating_income=300, revenue=1000)
    risk_manager.data_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})

    z = risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    assert z == 3.19


def test_z_score_grey_zone(risk_manager):
    # A=0.3, B=0.6, C=0.15, E=0.8 -> z=1.2*0.3+1.4*0.6+3.3*0.15+1.0*0.8=2.5 (1.81~2.99 灰色地帶)
    bs = make_balance_sheet('2024-Q1', current_assets=500, current_liabilities=200,
                             total_assets=1000, equity=600, retained_earnings=600)
    fs = make_financial_statement('2024-Q1', operating_income=150, revenue=800)
    risk_manager.data_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})

    z = risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    assert z == 2.5


def test_z_score_high_risk_zone(risk_manager):
    # A=-0.05, B=-0.05, C=-0.02, E=0.1 -> z=1.2*-0.05+1.4*-0.05+3.3*-0.02+1.0*0.1=-0.1 (<=1.81 高風險)
    bs = make_balance_sheet('2024-Q1', current_assets=150, current_liabilities=200,
                             total_assets=1000, equity=100, retained_earnings=-50)
    fs = make_financial_statement('2024-Q1', operating_income=-20, revenue=100)
    risk_manager.data_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})

    z = risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    assert z == -0.1


def test_z_score_falls_back_to_equity_when_retained_earnings_missing(risk_manager):
    # 沒有 RetainedEarnings 資料時，B 應改用 equity 計算，而非直接出錯
    bs = make_balance_sheet('2024-Q1', current_assets=500, current_liabilities=200,
                             total_assets=1000, equity=600, retained_earnings=None)
    fs = make_financial_statement('2024-Q1', operating_income=150, revenue=800)
    risk_manager.data_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})

    z = risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    # B 改用 equity/total_assets=0.6，其餘同 grey zone 案例 -> z=2.50
    assert z == 2.5


def test_z_score_uses_latest_quarter_only(risk_manager):
    # 資料含舊季度與最新季度，應只採用日期最大的那一季計算
    old_bs = make_balance_sheet('2023-Q4', current_assets=100, current_liabilities=900,
                                 total_assets=1000, equity=-800, retained_earnings=-900)
    latest_bs = make_balance_sheet('2024-Q1', current_assets=500, current_liabilities=200,
                                    total_assets=1000, equity=700, retained_earnings=600)
    bs = pd.concat([old_bs, latest_bs], ignore_index=True)

    old_fs = make_financial_statement('2023-Q4', operating_income=-900, revenue=10)
    latest_fs = make_financial_statement('2024-Q1', operating_income=300, revenue=1000)
    fs = pd.concat([old_fs, latest_fs], ignore_index=True)

    risk_manager.data_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})

    z = risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    # 若誤用舊季度資料，z 會是極端負值；正確應與 safe zone 案例一致 (3.19)
    assert z == 3.19


def test_z_score_passes_correct_args_to_data_service(risk_manager):
    bs = make_balance_sheet('2024-Q1', current_assets=500, current_liabilities=200,
                             total_assets=1000, equity=700, retained_earnings=600)
    fs = make_financial_statement('2024-Q1', operating_income=300, revenue=1000)
    fake_service = FakeDataService({'balance_sheet': bs, 'financial_statement': fs})
    risk_manager.data_service = fake_service

    risk_manager.z_score('2330', '2024-01-01', '2024-03-31')

    assert fake_service.calls == [
        ('2330', 'balance_sheet', '2024-01-01', '2024-03-31'),
        ('2330', 'financial_statement', '2024-01-01', '2024-03-31'),
    ]
