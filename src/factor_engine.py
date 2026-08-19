"""建立日期對齊的統一因子資料表，供策略研究與回測使用。"""

from __future__ import annotations

import pandas as pd


class FactorEngine:
    """在不修改來源資料的前提下，建立基本面與技術面因子。

    基本面資料會依實際可用日向後合併：已公告 ROE 持續有效到下一份資料
    出現，且不會把未來資訊帶入過去交易日。
    """

    PRICE_COLUMNS = {"date", "open", "close", "min", "max"}
    ROE_COLUMNS = {"date", "ROE"}

    def build_factor_table(
        self,
        stock_id: str,
        price_df: pd.DataFrame,
        roe_df: pd.DataFrame | None = None,
        *,
        ma_short_window: int = 5,
        ma_long_window: int = 20,
        rsi_period: int = 14,
        kd_period: int = 9,
    ) -> pd.DataFrame:
        """回傳每個交易日一列的基本面與技術面因子。

        ``roe_df`` 需使用 ``DataService`` 處理後的公告／生效日期，並包含
        ``date`` 與 ``ROE`` 欄位。未提供時仍可建立技術因子，ROE 則為空值。
        """
        self._require_columns(price_df, self.PRICE_COLUMNS, "price_df")
        self._validate_windows(
            ma_short_window, ma_long_window, rsi_period, kd_period
        )

        # 排序並保留每個交易日最後一筆資料，確保時間序列計算可重現。
        price = price_df.copy()
        price["date"] = pd.to_datetime(price["date"])
        price = price.sort_values("date").drop_duplicates("date", keep="last")
        price = price.reset_index(drop=True)

        factors = price[["date", "open", "close"]].copy()
        factors.insert(1, "code", str(stock_id))

        # 技術指標使用當日收盤價計算；交易執行端應延後至下一交易日開盤。
        factors["factor_MA_short"] = price["close"].rolling(ma_short_window).mean()
        factors["factor_MA_long"] = price["close"].rolling(ma_long_window).mean()
        factors["factor_RSI"] = self._rsi(price["close"], rsi_period)

        kd = self._kd(price, kd_period)
        factors["factor_KD_K"] = kd["K"]
        factors["factor_KD_D"] = kd["D"]
        # 僅合併當日或更早已公告的 ROE，避免前視偏誤。
        factors["factor_ROE"] = self._align_roe(factors, roe_df)

        factors["date"] = factors["date"].dt.strftime("%Y-%m-%d")
        return factors[
            [
                "date", "code", "open", "close", "factor_ROE",
                "factor_MA_short", "factor_MA_long", "factor_RSI",
                "factor_KD_K", "factor_KD_D",
            ]
        ]

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """以簡單移動平均計算 RSI。"""
        delta = close.diff()
        average_gain = delta.clip(lower=0).rolling(period).mean()
        average_loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _kd(price: pd.DataFrame, period: int) -> pd.DataFrame:
        """以 RSV 遞推計算 KD，初始 K、D 皆為 50。"""
        lowest = price["min"].rolling(period).min()
        highest = price["max"].rolling(period).max()
        rsv = (price["close"] - lowest) / (highest - lowest) * 100

        k_values: list[float] = []
        d_values: list[float] = []
        previous_k = previous_d = 50.0
        # 資料尚未滿足週期時保留空值，完成後才開始遞推。
        for value in rsv:
            if pd.isna(value):
                k_values.append(float("nan"))
                d_values.append(float("nan"))
                continue
            previous_k = previous_k * 2 / 3 + value / 3
            previous_d = previous_d * 2 / 3 + previous_k / 3
            k_values.append(previous_k)
            d_values.append(previous_d)
        return pd.DataFrame({"K": k_values, "D": d_values}, index=price.index)

    def _align_roe(
        self, factors: pd.DataFrame, roe_df: pd.DataFrame | None
    ) -> pd.Series:
        """將已公告的 ROE 向後對齊到每個交易日。"""
        if roe_df is None or roe_df.empty:
            return pd.Series(float("nan"), index=factors.index)

        self._require_columns(roe_df, self.ROE_COLUMNS, "roe_df")
        roe = roe_df[["date", "ROE"]].copy()
        roe["date"] = pd.to_datetime(roe["date"])
        roe = roe.sort_values("date").drop_duplicates("date", keep="last")

        # backward 只選擇當日或過去最近一次公告，不引用未來 ROE。
        aligned = pd.merge_asof(
            factors[["date"]].sort_values("date"), roe,
            on="date", direction="backward",
        )
        return aligned["ROE"]

    @staticmethod
    def _require_columns(
        df: pd.DataFrame, required: set[str], argument_name: str
    ) -> None:
        """確認輸入資料具備計算所需欄位。"""
        missing = required.difference(df.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{argument_name} is missing required columns: {names}")

    @staticmethod
    def _validate_windows(*windows: int) -> None:
        """確認技術指標期間皆為正整數。"""
        if any(not isinstance(window, int) or window <= 0 for window in windows):
            raise ValueError("all factor windows must be positive integers")
