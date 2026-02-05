"""
TextProcessor：文字處理器
功能：將 DataFrame 轉換成自然語言文字片段，供 RAG 使用
"""

import pandas as pd
import numpy as np


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
        "CashAndCashEquivalents": "現金及約當現金",
        "AccountsReceivableNet": "應收帳款淨額",
        "Inventories": "存貨",
        "CurrentAssets": "流動資產",
        "PropertyPlantAndEquipment": "不動產廠房設備",
        "TotalAssets": "總資產",
        "ShorttermBorrowings": "短期借款" ,
        "LongtermBorrowings": "長期借款",
        "CurrentLiabilities": "流動負債",
        "Liabilities": "負債總額",
        "CapitalStock": "股本",
        "RetainedEarnings": "保留盈餘",
        "Equity": "股東權益"
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

    # =========================================================================
    # _format_row：格式化單一 row（私有方法）
    # =========================================================================
    # 注意：這是 class 的方法，和 _date_to_quarter、_format_number 同層級
    #       不是寫在 df_to_chunks 裡面
    #
    # 對標 C++：
    #   class TextProcessor {
    #   private:
    #       string _format_row(const Row& row, const string& stock_name);
    #   };
    # =========================================================================
    def _format_row(self, row, stock_name: str) -> str:
        """
        格式化單一 row 為文字

        Args:
            row: DataFrame 的一行資料（由 .apply() 傳入）
            stock_name: 股票名稱

        Returns:
            格式化後的文字，例如 "2023年第1季 台積電(2330) 的EPS為 7.98 元。"
        """
        # 轉換日期 → 季度
        quarter_str = self._date_to_quarter(row['date'])

        # 查表：英文指標 → 中文名稱
        indicator_name = self.INDICATOR_NAMES[row['type']]

        # 格式化數字
        value_str = self._format_number(row['value'], row['type'])

        # 組合股票名稱
        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"

        return f"{quarter_str} {stock_str} 的{indicator_name}為 {value_str}。"

    # =========================================================================
    # df_to_chunks：將 DataFrame 轉換成文字片段列表（公開方法）
    # =========================================================================
    # 重構前：用 for + iterrows() 逐行處理（像 C++ 的 for 迴圈）
    # 重構後：用 .apply() 向量化處理（Pandas 的慣用寫法，底層用 C 執行更快）
    #
    # 對標 C++：
    #   重構前：for (auto& row : df) { chunks.push_back(format(row)); }
    #   重構後：std::transform(df.begin(), df.end(), chunks.begin(), format);
    # =========================================================================
    def df_to_chunks(self, df: pd.DataFrame, stock_name: str = None) -> list[str]:
        """
        將 DataFrame 轉換成文字片段列表

        Args:
            df: 財報 DataFrame
            stock_name: 股票名稱（可選）

        Returns:
            文字片段列表
        """
        # 1. 去除重複資料（同日期、同股票、同指標只保留一筆）
        df = df.drop_duplicates(subset=["date", "stock_id", "type"])

        # 2. 篩選我們關心的指標
        target_types = list(self.INDICATOR_NAMES.keys())
        filtered_df = df[df['type'].isin(target_types)]

        # 3. 用 .apply() 對每一行執行 _format_row，產生文字
        #
        #    拆解這行：
        #    filtered_df.apply(...)  → 對每一行執行函式，回傳 Series
        #    lambda row: ...        → 匿名函式，把 row 傳給 _format_row
        #    axis=1                 → 逐「行」處理（axis=0 是逐「列」）
        #    .tolist()              → 把 Series 轉成 Python list
        #
        #    對標 C++ lambda：
        #    auto func = [&](const Row& row) { return _format_row(row, stock_name); };
        chunks = filtered_df.apply(
            lambda row: self._format_row(row, stock_name), axis=1
        ).tolist()

        return chunks
    

    def _format_per_row(self, row, stock_name: str) -> str:
        """
        格式化單一 row 為文字

        Args:
            row: DataFrame 的一行資料（由 .apply() 傳入）
            stock_name: 股票名稱

        Returns:
            格式化後的文字，例如 "2023年第1季 台積電(2330) 的EPS為 7.98 元。"
        """
       
        # 組合股票名稱
        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"

        return f"{row['date']}{stock_str}本益比{row['PER']}倍，殖利率{row['dividend_yield']}%，股價淨值比{row['PBR']}倍。"


    def per_to_chunks(self, df: pd.DataFrame, stock_name: str = None):
        return df.apply(lambda row: self._format_per_row(row, stock_name), axis=1
                    ).tolist()


    def _calculate_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df['year'] = df['date'].str[:4].astype(int)
        df['month'] = df['date'].str[5:7].astype(int)
        df['quarter'] = df['month'].map(self.MONTH_TO_QUARTER)
        df = df.sort_values(['stock_id', 'type', 'year', 'quarter'])
        df['prev_value'] = df.groupby(['stock_id', 'type'])['value'].shift(1)
        df['qoq'] = (df['value'] - df['prev_value']) / abs(df['prev_value']) * 100
        df['qoq'] = df['qoq'].replace((np.inf, -np.inf), np.nan)

        df['prev_year_value'] = df.groupby(['stock_id', 'type'])['value'].shift(4)
        df['yoy'] = (df['value'] - df['prev_year_value']) / abs(df['prev_year_value']) * 100
        df['yoy'] = df['yoy'].replace((np.inf, -np.inf), np.nan)

        return df
    


    def _format_growth_row(self, row, stock_name: str) -> str:
        indicator_name = self.INDICATOR_NAMES[row['type']]

        # 格式化數字
        value_str = self._format_number(row['value'], row['type'])
        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"

        if pd.isna(row['qoq']) and pd.notna(row['yoy']):
            return f"{row['year']}年第{row['quarter']}季 {stock_str} 的{indicator_name}為 {value_str}，季增[無資料]，年增{row['yoy']:.2f}%。"
        elif pd.notna(row['qoq']) and pd.isna(row['yoy']):
            return f"{row['year']}年第{row['quarter']}季 {stock_str} 的{indicator_name}為 {value_str}，季增{row['qoq']:.2f}%，年增[無資料]。"
        elif pd.isna(row['qoq']) and pd.isna(row['yoy']):
            return f"{row['year']}年第{row['quarter']}季 {stock_str} 的{indicator_name}為 {value_str}，季增[無資料]，年增[無資料]。"
        else:
            return f"{row['year']}年第{row['quarter']}季 {stock_str} 的{indicator_name}為 {value_str}，季增{row['qoq']:.2f}%，年增{row['yoy']:.2f}%。"
        


    def growth_to_chunks(self, df: pd.DataFrame, stock_name: str = None) -> list[str]:
        df = self._calculate_growth_rates(df)
        target_types = list(self.INDICATOR_NAMES.keys())
        filtered_df = df[df['type'].isin(target_types)]
        
        chunks = filtered_df.apply(
            lambda row: self._format_growth_row(row, stock_name), axis=1
        ).tolist()

        return chunks
    

    def roe_roa_to_chunks(self, financial_df: pd.DataFrame, balance_df: pd.DataFrame, stock_name: str = None) -> list[str]:
        assets = balance_df[balance_df['type'] == 'TotalAssets']
        equity = balance_df[balance_df['type'] == 'Equity']
        incomeAfterTaxes = financial_df[financial_df['type'] == 'IncomeAfterTaxes']

        incomeAfterTaxes = incomeAfterTaxes[['stock_id', 'date', 'value']].rename(columns={'value': 'net_income'})
        equity = equity[['stock_id', 'date', 'value']].rename(columns={'value': 'equity'})
        assets = assets[['stock_id', 'date', 'value']].rename(columns={'value': 'total_assets'})

        incomeAfterTaxes['stock_id'] = incomeAfterTaxes['stock_id'].astype(str)
        equity['stock_id'] = equity['stock_id'].astype(str)
        assets['stock_id'] = assets['stock_id'].astype(str)


        df = pd.merge(incomeAfterTaxes, equity, on=['stock_id', 'date'], how= 'inner')
        df = pd.merge(df, assets, on=['stock_id', 'date'], how= 'inner')
        df = self._roe_roa_calculation(df) 

        chunks = df.apply(lambda row: self._format_roe_roa_row(row, stock_name), axis=1).tolist()
        return chunks
    

    def _roe_roa_calculation(self, df: pd.DataFrame):
        df['roe'] = df['net_income'] / df['equity'] * 100
        df['roa'] = df['net_income'] / df['total_assets'] * 100
        return df
    


    def _format_roe_roa_row(self, row, stock_name: str) -> str:
        quarter_str = self._date_to_quarter(row['date'])

        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"
        
        return f"{quarter_str}{stock_str}的ROE為{row['roe']:.2f}%，ROA為{row['roa']:.2f}%"



    def debt_current_to_chunks(self, balance_df: pd.DataFrame, stock_name: str = None) -> list[str]:
        liabilities = balance_df[balance_df['type'] == 'Liabilities']
        total_assets = balance_df[balance_df['type'] == 'TotalAssets']
        current_assets = balance_df[balance_df['type'] == 'CurrentAssets']
        current_liabilities = balance_df[balance_df['type'] == 'CurrentLiabilities']

        liabilities = liabilities[['stock_id','date', 'value']].rename(columns={'value': 'liabilities'})
        total_assets = total_assets[['stock_id','date', 'value']].rename(columns={'value': 'total_assets'})
        current_assets = current_assets[['stock_id','date', 'value']].rename(columns={'value': 'current_assets'})
        current_liabilities = current_liabilities[['stock_id','date', 'value']].rename(columns={'value': 'current_liabilities'})

        liabilities['stock_id'] = liabilities['stock_id'].astype(str)
        total_assets['stock_id'] = total_assets['stock_id'].astype(str)
        current_assets['stock_id'] = current_assets['stock_id'].astype(str)
        current_liabilities['stock_id'] = current_liabilities['stock_id'].astype(str)

                                                                                       
        df = pd.merge(liabilities, total_assets, on=['stock_id', 'date'], how= 'inner')
        df = pd.merge(df, current_assets, on=['stock_id', 'date'], how= 'inner')
        df = pd.merge(df, current_liabilities, on=['stock_id', 'date'], how= 'inner')

        df = self._debt_calculation(df) 

        chunks = df.apply(lambda row: self._format_debt_row(row, stock_name), axis=1).tolist()
        return chunks

    def _debt_calculation(self, df: pd.DataFrame):
        df['debtRatio'] = df['liabilities'] / df['total_assets'] * 100
        df['currentRatio'] = df['current_assets'] / df['current_liabilities'] * 100
        return df
    


    def _format_debt_row(self, row, stock_name: str) -> str: 
        quarter_str = self._date_to_quarter(row['date'])

        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"
        
        return f"{quarter_str}{stock_str}的負債比為{row['debtRatio']:.2f}%流動比率為{row['currentRatio']:.2f}%"


    def margin_to_chunks(self, financial_df: pd.DataFrame, stock_name: str = None) -> list[str]:
        """計算毛利率和營益率，回傳 chunks"""
        gross_profit = financial_df[financial_df['type'] == 'GrossProfit']
        operating_income = financial_df[financial_df['type'] == 'OperatingIncome']
        revenue = financial_df[financial_df['type'] == 'Revenue']

        gross_profit = gross_profit[['stock_id', 'date', 'value']].rename(columns={'value': 'gross_profit'})
        operating_income = operating_income[['stock_id', 'date', 'value']].rename(columns={'value': 'operating_income'})
        revenue = revenue[['stock_id', 'date', 'value']].rename(columns={'value': 'revenue'})

        gross_profit['stock_id'] = gross_profit['stock_id'].astype(str)
        operating_income['stock_id'] = operating_income['stock_id'].astype(str)
        revenue['stock_id'] = revenue['stock_id'].astype(str)

        df = pd.merge(gross_profit, operating_income, on=['stock_id', 'date'], how='inner')
        df = pd.merge(df, revenue, on=['stock_id', 'date'], how='inner')

        df = self._calculate_margin(df)

        chunks = df.apply(lambda row: self._format_margin_row(row, stock_name), axis=1).tolist()
        return chunks

    def _calculate_margin(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算毛利率和營益率"""
        df['gross_margin'] = df['gross_profit'] / df['revenue'] * 100
        df['operating_margin'] = df['operating_income'] / df['revenue'] * 100
        return df

    def _format_margin_row(self, row, stock_name: str) -> str:
        """格式化毛利率/營益率輸出"""
        quarter_str = self._date_to_quarter(row['date'])

        if stock_name:
            stock_str = f"{stock_name}({row['stock_id']})"
        else:
            stock_str = f"股票{row['stock_id']}"

        return f"{quarter_str}{stock_str}的毛利率為{row['gross_margin']:.2f}%，營益率為{row['operating_margin']:.2f}%"