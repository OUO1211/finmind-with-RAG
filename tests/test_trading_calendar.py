from datetime import date

import pytest

from src.trading_calendar import TradingCalendar


def test_next_trading_day_uses_the_dates_in_the_csv_not_weekdays(tmp_path):
    # 9/1 為平日但不在日曆中，下一個可交易日必須是 9/2。
    calendar_path = tmp_path / "trading_calendar.csv"
    calendar_path.write_text("date\n2026-08-28\n2026-08-31\n2026-09-02\n", encoding="utf-8")

    calendar = TradingCalendar.load(calendar_path)

    assert calendar.next_after(date(2026, 8, 31)) == date(2026, 9, 2)


def test_calendar_rejects_missing_date_column(tmp_path):
    calendar_path = tmp_path / "trading_calendar.csv"
    calendar_path.write_text("session\n2026-08-31\n", encoding="utf-8")

    with pytest.raises(ValueError, match="date"):
        TradingCalendar.load(calendar_path)


def test_calendar_rejects_a_date_without_a_following_session(tmp_path):
    calendar_path = tmp_path / "trading_calendar.csv"
    calendar_path.write_text("date\n2026-08-31\n", encoding="utf-8")

    with pytest.raises(ValueError, match="next trading day"):
        TradingCalendar.load(calendar_path).next_after(date(2026, 8, 31))
