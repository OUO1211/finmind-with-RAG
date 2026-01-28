"""
TextProcessor：文字處理器
功能：將 DataFrame 轉換成自然語言文字片段，供 RAG 使用
"""

import pandas as pd


class TextProcessor:
    """
    文字處理器

    職責：
    1. 將財報 DataFrame 轉換成自然語言句子
    2. 日期轉換成季度格式
    3. 數字格式化（大數字轉成億元）
    """

    # 指標名稱對照表（英文 → 中文）
    INDICATOR_NAMES = {
        "EPS": "每股盈餘(EPS)",
        "Revenue": "營收",
        "GrossProfit": "毛利",
        "OperatingIncome": "營業利益",
        "PreTaxIncome": "稅前淨利",
        "NetIncome": "淨利",
        "OperatingExpenses": "營業費用",
    }

    # 月份 → 季度對照表
    MONTH_TO_QUARTER = {
        3: 1,   # Q1
        6: 2,   # Q2
        9: 3,   # Q3
        12: 4,  # Q4
    }

    def _date_to_quarter(self, date_str: str) -> str:
        """將日期字串轉換成季度格式"""
        parts = date_str.split("-")
        year = parts[0]
        month = int(parts[1])
        quarter = self.MONTH_TO_QUARTER.get(month, 1)
        return f"{year}年第{quarter}季"

    def _format_number(self, value: float, indicator_type: str) -> str:
        """格式化數字，讓大數字更易讀"""
        if indicator_type == "EPS":
            return f"{value:.2f} 元"

        if abs(value) >= 1e8:
            value_in_billion = value / 1e8
            return f"{value_in_billion:,.2f} 億元"
        elif abs(value) >= 1e4:
            value_in_ten_thousand = value / 1e4
            return f"{value_in_ten_thousand:,.2f} 萬元"
        else:
            return f"{value:,.2f} 元"

    def df_to_chunks(self, df: pd.DataFrame, stock_name: str = None) -> list[str]:
        """
        將 DataFrame 轉換成文字片段列表

        Args:
            df: 財報 DataFrame
            stock_name: 股票名稱（可選）

        Returns:
            文字片段列表
        """
        chunks = []

        target_types = list(self.INDICATOR_NAMES.keys())
        filtered_df = df[df['type'].isin(target_types)]

        for _, row in filtered_df.iterrows():
            date_str = row['date']
            stock_id = row['stock_id']
            indicator_type = row['type']
            value = row['value']

            quarter_str = self._date_to_quarter(date_str)
            indicator_name = self.INDICATOR_NAMES[indicator_type]
            value_str = self._format_number(value, indicator_type)

            if stock_name:
                stock_str = f"{stock_name}({stock_id})"
            else:
                stock_str = f"股票{stock_id}"

            chunk = f"{quarter_str} {stock_str} 的{indicator_name}為 {value_str}。"
            chunks.append(chunk)

        return chunks
