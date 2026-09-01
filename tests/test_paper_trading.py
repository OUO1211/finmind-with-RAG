from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from hypothesis import given, strategies as st
from src.forward_test_spec import ForwardTestSpec
from src.paper_trading import PaperTradingService

SPEC_PATH = "configs/forward_test_v1.yaml"
pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


def _factors(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "code", "factor_KD_K", "factor_KD_D"])


def test_close_generates_kd_orders_for_the_next_trading_day(tmp_path):
    # Given：1111 在收盤時出現 KD 黃金交叉，且位於當日可交易股票池。
    service = PaperTradingService(tmp_path, ForwardTestSpec.load(SPEC_PATH))
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
        ("2026-08-28", "2222", 60, 50),
        ("2026-08-29", "2222", 40, 50),
    ])

    # When：收盤後產生訂單；週末後的第一個交易日為 8/31。
    orders = service.generate_close_orders(
        date(2026, 8, 29), factors, {"1111"},
        [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)],
    )

    # Then：只建立 1111 的買單，並以 pending 狀態持久化保存。
    assert [(order.code, order.side, order.execution_date) for order in orders] == [
        ("1111", "buy", date(2026, 8, 31))
    ]
    saved = pd.read_csv(tmp_path / "orders.csv")
    assert saved.loc[0, "status"] == "pending"
    assert saved.loc[0, "signal_date"] == "2026-08-29"


def test_open_executes_sells_before_buys_and_records_cost_inclusive_fill(tmp_path):
    # Given：先以黃金交叉買入 1111，使其成為既有持倉。
    spec = ForwardTestSpec.load(SPEC_PATH)
    service = PaperTradingService(tmp_path, spec)
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
        ("2026-08-31", "1111", 60, 50),
        ("2026-09-01", "1111", 40, 50),
        ("2026-08-31", "2222", 40, 50),
        ("2026-09-01", "2222", 60, 50),
    ])
    calendar = [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2)]
    service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, calendar)
    first_fills = service.execute_open(date(2026, 8, 31), {"1111": 100.0})

    # Then：買進成交價包含不利滑價，持倉數量沿用既有回測成本模型。
    assert first_fills[0].fill_price == pytest.approx(100.1)
    assert first_fills[0].shares == pytest.approx(1_000_000 / 5 / (100.1 * 1.001425))

    # When：1111 出現死亡交叉、2222 出現黃金交叉，兩筆訂單同日開盤執行。
    service.generate_close_orders(date(2026, 9, 1), factors, {"2222"}, calendar)
    fills = service.execute_open(date(2026, 9, 2), {"1111": 120.0, "2222": 80.0})

    # Then：必須先賣後買，賣出所得可在同一開盤用於新部位。
    assert [fill.side for fill in fills] == ["sell", "buy"]
    assert fills[0].fill_price == pytest.approx(120 * (1 - spec.slippage_rate))
    assert service.load_state().positions.keys() == {"2222"}
    assert len(pd.read_csv(tmp_path / "fills.csv")) == 3


def test_repeated_close_or_open_run_is_idempotent(tmp_path):
    # Given：同一個收盤訊號可能因排程重試而重複執行。
    service = PaperTradingService(tmp_path, ForwardTestSpec.load(SPEC_PATH))
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
    ])
    calendar = [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)]

    first = service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, calendar)
    second = service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, calendar)
    service.execute_open(date(2026, 8, 31), {"1111": 100.0})
    repeated_fills = service.execute_open(date(2026, 8, 31), {"1111": 100.0})

    # Then：不得重複建立訂單或重複成交。
    assert len(first) == len(second) == 1
    assert repeated_fills == []
    assert len(pd.read_csv(tmp_path / "orders.csv")) == 1


def test_missing_opening_price_cancels_the_order_without_changing_state(tmp_path):
    # Given：收盤時已建立次日開盤的買單。
    service = PaperTradingService(tmp_path, ForwardTestSpec.load(SPEC_PATH))
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
    ])
    calendar = [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)]
    service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, calendar)

    # When：到期開盤卻沒有該股票的有效開盤價。
    fills = service.execute_open(date(2026, 8, 31), {})

    # Then：訂單取消，且不產生成交、不影響現金與持倉。
    assert fills == []
    assert service.load_state().cash == pytest.approx(1_000_000)
    assert service.load_state().positions == {}
    assert pd.read_csv(tmp_path / "orders.csv").loc[0, "status"] == "cancelled"


def test_close_requires_a_following_trading_day(tmp_path):
    # 沒有下一個交易日時不可建立無法執行的紙上訂單。
    service = PaperTradingService(tmp_path, ForwardTestSpec.load(SPEC_PATH))
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
    ])

    with pytest.raises(ValueError, match="next trading day"):
        service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, [date(2026, 8, 29)])


@given(opening_price=st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False, allow_infinity=False))
def test_first_buy_preserves_cash_position_and_cost_invariants(tmp_path_factory, opening_price):
    """任意正開盤價下，首次買進都不可超支或超過部位上限。"""
    
    ledger_dir = tmp_path_factory.mktemp("paper_trading_property")
    spec = ForwardTestSpec.load(SPEC_PATH)
    service = PaperTradingService(ledger_dir, spec)
    factors = _factors([
        ("2026-08-28", "1111", 40, 50),
        ("2026-08-29", "1111", 60, 50),
    ])
    calendar = [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)]

    service.generate_close_orders(date(2026, 8, 29), factors, {"1111"}, calendar)
    fills = service.execute_open(date(2026, 8, 31), {"1111": opening_price})
    state = service.load_state()

    # 首次買進只使用五個等權空缺槽中的一個，且價格不影響預先配置的現金額。
    capital_used = spec.initial_capital / spec.max_positions
    expected_cost_per_share = opening_price * (1 + spec.slippage_rate) * (1 + spec.commission_rate)
    assert len(fills) == 1
    assert 0 <= state.cash == pytest.approx(spec.initial_capital - capital_used)
    assert 1 <= len(state.positions) <= spec.max_positions
    assert fills[0].fill_price == pytest.approx(opening_price * (1 + spec.slippage_rate))
    assert fills[0].shares * expected_cost_per_share == pytest.approx(capital_used)


@given(opening_prices=st.lists(
    st.floats(min_value=0.01, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    min_size=6,
    max_size=50,
))
def test_simultaneous_kd_buys_respect_portfolio_limit_and_cancel_excess_orders(
    tmp_path_factory, opening_prices,
):
    """同日大量 KD 買進訊號不會超過持倉與現金限制。"""
    ledger_dir = tmp_path_factory.mktemp("simultaneous_kd_buys")
    spec = ForwardTestSpec.load(SPEC_PATH)
    service = PaperTradingService(ledger_dir, spec)
    codes = [f"{index:04d}" for index in range(1, len(opening_prices) + 1)]
    factors = _factors([
        row
        for code in codes
        for row in [
            ("2026-08-28", code, 40, 50),
            ("2026-08-29", code, 60, 50),
        ]
    ])
    calendar = [date(2026, 8, 28), date(2026, 8, 29), date(2026, 8, 31)]

    service.generate_close_orders(date(2026, 8, 29), factors, codes, calendar)
    service.execute_open(date(2026, 8, 31), dict(zip(codes, opening_prices)))

    state = service.load_state()
    orders = pd.read_csv(ledger_dir / "orders.csv")
    excess_orders = orders.iloc[spec.max_positions:]
    assert len(state.positions) <= spec.max_positions
    assert state.cash >= 0
    assert (excess_orders["status"] == "cancelled").all()
