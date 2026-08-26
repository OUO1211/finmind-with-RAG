from pathlib import Path

import pytest

from src.forward_test_spec import ForwardTestSpec


SPEC_PATH = Path("configs/forward_test_v1.yaml")


def test_v1_spec_locks_the_agreed_strategy_and_costs():
    spec = ForwardTestSpec.load(SPEC_PATH)

    assert spec.universe_max_stocks == 50
    assert spec.kd_period == 9
    assert spec.max_positions == 5
    assert spec.commission_rate == pytest.approx(0.001425)
    assert spec.sell_tax_rate == pytest.approx(0.003)
    assert spec.slippage_rate == pytest.approx(0.001)


def test_v1_spec_creates_the_same_cost_model_used_in_backtests():
    backtester = ForwardTestSpec.load(SPEC_PATH).make_backtester()

    assert backtester.slippage_rate == pytest.approx(0.001)
    assert backtester.commission_rate == pytest.approx(0.001425)
    assert backtester.tax_rate == pytest.approx(0.003)


def test_v1_spec_rejects_a_changed_kd_period(tmp_path):
    changed = tmp_path / "changed.yaml"
    changed.write_text(SPEC_PATH.read_text(encoding="utf-8").replace("period: 9", "period: 14"), encoding="utf-8")

    with pytest.raises(ValueError, match="period 9"):
        ForwardTestSpec.load(changed)
