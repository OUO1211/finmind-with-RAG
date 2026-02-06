"""
DataService：資料服務層
功能：整合 CacheManager + DataFetcher，提供統一的資料獲取介面
"""

import pandas as pd
from .cache_manager import CacheManager
from .data_fetcher import DataFetcher


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

        # Step 3: 存入快取
        if not df.empty:
            self.cache.set(stock_id, data_type, start_date, end_date, df)

        return df

    def get_supported_types(self) -> list:
        """取得支援的資料類型列表"""
        return list(self._fetch_methods.keys())
