"""A local, explicit list of Taiwan market trading sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    """Trading sessions loaded from a version-controlled CSV file."""

    dates: tuple[date, ...]

    @classmethod
    def load(cls, path: str | Path) -> "TradingCalendar":
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise ValueError("trading calendar must contain a date column")
        try:
            dates = tuple(sorted(set(pd.to_datetime(frame["date"], errors="raise").dt.date)))
        except (TypeError, ValueError) as error:
            raise ValueError("trading calendar contains an invalid date") from error
        if not dates:
            raise ValueError("trading calendar must contain at least one date")
        return cls(dates)

    def next_after(self, session: date) -> date:
        for candidate in self.dates:
            if candidate > session:
                return candidate
        raise ValueError("a next trading day is required")

    def __iter__(self):
        return iter(self.dates)
