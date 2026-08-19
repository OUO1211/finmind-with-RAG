"""
DataService：資料服務層
功能：整合 CacheManager + DataFetcher，提供統一的資料獲取介面
"""

import numpy as np
import pandas as pd
from .cache_manager import CacheManager
from .data_fetcher import DataFetcher

# 財報/月營收都是「期間結束後才會公告」，不能讓回測在期間結束當天就看到這筆資料。
# 這裡改用法規規定的「最晚公告期限」（固定行事曆日期）當作生效日期，即使實際公告
# 可能更早，用最晚期限保證絕對不會誤用還沒公開的資訊（安全邊界，不是精確值）。
_QUARTERLY_DISCLOSURE_RULE = {
    # 季度結束月份 → (生效年份要不要 +1, 生效月, 生效日)
    3: (0, 5, 15),
    6: (0, 8, 14),
    9: (0, 11, 14),
    12: (1, 3, 31),
}
# 這些資料類型是「季度結束後才公告」，需要套用上面的固定行事曆規則；
# stock_price、per 本身就是逐日資料，沒有這個問題，不需要調整。
_QUARTERLY_TYPES = {"financial_statement", "balance_sheet", "cash_flow"}


class DataService:
    """
    資料服務層

    職責：
    1. 整合 CacheManager 和 DataFetcher
    2. 提供統一的 get_data() 介面
    3. 自動處理「先查快取 → 沒有才打 API → 存入快取」的流程
    """

    def __init__(self, cache_dir: str = "data", token: str = None):
        """
        建構子

        Args:
            cache_dir: 快取目錄
            token: FinMind API Token
        """
        self.cache = CacheManager(cache_dir)
        self.fetcher = DataFetcher(token)

        # 建立「資料類型 → API 方法」的對應表
        self._fetch_methods = {
            "financial_statement": self.fetcher.get_financial_statement,
            "stock_price": self.fetcher.get_stock_price,
            "monthly_revenue": self.fetcher.get_monthly_revenue,
            "balance_sheet": self.fetcher.get_balance_sheet,
            "per": self.fetcher.get_per,
            "cash_flow": self.fetcher.get_cash_flow_statement,
        }

    def get_data(self, stock_id: str, data_type: str,
                 start_date: str, end_date: str) -> pd.DataFrame:
        """
        統一的資料獲取介面

        Args:
            stock_id: 股票代號，如 "2330"
            data_type: 資料類型
            start_date: 起始日期
            end_date: 結束日期

        Returns:
            DataFrame：股票資料
        """
        # Step 1: 查快取
        df = self.cache.get(stock_id, data_type, start_date, end_date)
        if df is not None:
            return df

        # Step 2: 快取沒有，打 API
        if data_type not in self._fetch_methods:
            print(f"[DataService] 錯誤：不支援的資料類型 '{data_type}'")
            return pd.DataFrame()

        fetch_method = self._fetch_methods[data_type]
        df = fetch_method(stock_id, start_date, end_date)

        # Step 2.5：財報/月營收要把「期間結束日」換成「生效日期」，避免回測
        # 提前看到還沒公告的資料（防污染要在存入快取之前做，這樣快取存的
        # 就已經是安全可用的日期，之後不管誰讀快取都不用重算一次）
        df = self._shift_to_disclosure_date(df, data_type)

        # Step 3: 存入快取
        if not df.empty:
            self.cache.set(stock_id, data_type, start_date, end_date, df)

        return df

    def _shift_to_disclosure_date(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        把財報/月營收的「期間結束日」換成「法定最晚公告日」

        Args:
            df: 原始資料，'date' 欄位是期間結束日（例如季報 3/31、月營收所屬月份）
            data_type: 資料類型，決定套用哪一種公告時間規則

        Returns:
            DataFrame：'date' 欄位已替換成生效日期的資料（非財報類資料原樣傳回）
        """
        if df.empty or 'date' not in df.columns:
            return df

        df = df.copy()
        period_end = pd.to_datetime(df['date'])

        if data_type == 'monthly_revenue':
            # 月營收法規規定次月 10 日內公告，用「次月 10 日」當生效日期
            df['date'] = (period_end + pd.DateOffset(months=1)).dt.strftime('%Y-%m') + '-10'
            return df

        if data_type not in _QUARTERLY_TYPES:
            return df

        # np.select 是向量化的多分支條件判斷（概念上像逐列跑一串 if/elif），
        # 依季度結束月份（3/6/9/12）分別決定生效年份要不要 +1、生效月、生效日。
        # 非標準季度結束月份（理論上不會發生）則維持原日期，屬於防禦性寫法。
        months = period_end.dt.month
        years = period_end.dt.year

        conditions = [months == m for m in _QUARTERLY_DISCLOSURE_RULE]
        year_offsets = [rule[0] for rule in _QUARTERLY_DISCLOSURE_RULE.values()]
        target_months = [rule[1] for rule in _QUARTERLY_DISCLOSURE_RULE.values()]
        target_days = [rule[2] for rule in _QUARTERLY_DISCLOSURE_RULE.values()]

        effective_year = years + np.select(conditions, year_offsets, default=0)
        effective_month = np.select(conditions, target_months, default=months)
        effective_day = np.select(conditions, target_days, default=period_end.dt.day)

        # pd.to_datetime 可以直接吃 {'year':..., 'month':..., 'day':...} 這種欄位式
        # 字典，逐列組成 Timestamp——等同於用三個獨立欄位手動組出一個日期物件。
        df['date'] = pd.to_datetime({
            'year': effective_year, 'month': effective_month, 'day': effective_day
        }).dt.strftime('%Y-%m-%d')

        return df

    def get_supported_types(self) -> list:
        """取得支援的資料類型列表"""
        return list(self._fetch_methods.keys())
