"""
CacheManager：快取管理器
功能：檢查、讀取、寫入快取，支援不同資料類型的過期策略
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


class CacheManager:
    """
    快取管理器

    職責：
    1. 生成快取檔案路徑 (依據 stock_id, data_type, 日期範圍)
    2. 檢查快取是否存在且有效 (未過期)
    3. 讀取/寫入快取
    """

    # 類別常數：各資料類型的過期天數
    EXPIRY_DAYS = {
        "stock_price": 1,           # 日股價：每天更新
        "financial_statement": 90,  # 財報：每季更新
        "monthly_revenue": 30,      # 月營收：每月更新
    }

    def __init__(self, cache_dir: str = "data"):
        """
        建構子

        Args:
            cache_dir: 快取根目錄，預設為 "data"
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, stock_id: str, data_type: str,
                      start_date: str, end_date: str) -> Path:
        """生成快取檔案路徑"""
        type_dir = self.cache_dir / data_type
        type_dir.mkdir(exist_ok=True)
        filename = f"{stock_id}_{start_date}_{end_date}.csv"
        return type_dir / filename

    def _is_expired(self, file_path: Path, data_type: str) -> bool:
        """檢查快取是否過期"""
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        days_passed = (datetime.now() - mtime).days
        expiry = self.EXPIRY_DAYS.get(data_type, 1)
        return days_passed > expiry

    def get(self, stock_id: str, data_type: str,
            start_date: str, end_date: str) -> pd.DataFrame | None:
        """
        讀取快取

        Returns:
            DataFrame：快取存在且未過期
            None：快取不存在或已過期
        """
        file_path = self._generate_key(stock_id, data_type, start_date, end_date)

        if not file_path.exists():
            print(f"[Cache] 快取不存在：{file_path}")
            return None

        if self._is_expired(file_path, data_type):
            print(f"[Cache] 快取已過期：{file_path}")
            return None

        print(f"[Cache] 命中快取：{file_path}")
        return pd.read_csv(file_path)

    def set(self, stock_id: str, data_type: str,
            start_date: str, end_date: str, df: pd.DataFrame) -> None:
        """寫入快取"""
        file_path = self._generate_key(stock_id, data_type, start_date, end_date)
        df.to_csv(file_path, index=False)
        print(f"[Cache] 已寫入快取：{file_path}")
