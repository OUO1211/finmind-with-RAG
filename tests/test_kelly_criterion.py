import pytest
from src.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    # fixture：每個測試都會拿到一個乾淨的 RiskManager 實例
    return RiskManager()


def test_kelly_criterion_no_trades_returns_zero(risk_manager):
    # 沒有交易 -> win_rate() 回傳 profit_loss_ratio = 0 -> 函式提早 return 0
    result = risk_manager.kelly_criterion([])

    assert result == 0


def test_kelly_criterion_all_wins_returns_zero(risk_manager):
    # 全部獲利，沒有虧損交易 -> profit_loss_ratio 仍然是 0（同上分支）
    trades = [
        {'return': 10},
        {'return': 5},
    ]

    result = risk_manager.kelly_criterion(trades)

    assert result == 0


def test_kelly_criterion_all_losses_returns_zero(risk_manager):
    # 全部虧損，沒有獲利交易 -> profit_loss_ratio 仍然是 0（同上分支）
    trades = [
        {'return': -5},
        {'return': -3},
    ]

    result = risk_manager.kelly_criterion(trades)

    assert result == 0


def test_kelly_criterion_mixed_win_loss(risk_manager):
    # 2 勝 (+20%) / 2 敗 (-10%)
    # win_rate = 50%, avg_profit = 20, avg_loss = -10, profit_loss_ratio(b) = 2.0
    # p = 0.5, q = 0.5
    # kelly = (b*p - q) / b = (2*0.5 - 0.5) / 2 = 0.25
    # half_kelly = kelly / 2 = 0.125
    trades = [
        {'return': 20},
        {'return': 20},
        {'return': -10},
        {'return': -10},
    ]

    result = risk_manager.kelly_criterion(trades)

    assert result == pytest.approx(0.125)
