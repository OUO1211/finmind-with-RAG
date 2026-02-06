"""
DataFetcher：資料獲取器
功能：負責呼叫 FinMind API 獲取各類股票資料
"""

from FinMind.data import DataLoader
import pandas as pd
import requests
import os


class DataFetcher:
    """
    資料獲取器

    職責：
    1. 管理 FinMind API Token
    2. 呼叫各種 FinMind API
    3. 錯誤處理（API 失敗時回傳空 DataFrame）
    """

    def __init__(self, token: str = None):
        """
        建構子

        Args:
            token: FinMind API Token，若不傳則從環境變數讀取
        """
        self.token = token or os.getenv("FINMIND_TOKEN")
        self.loader = DataLoader()

        if self.token:
            self.loader.login_by_token(api_token=self.token)
            print("[DataFetcher] 已使用 Token 登入（600次/小時）")
        else:
            print("[DataFetcher] 未設定 Token，使用匿名模式（300次/小時）")

    def get_financial_statement(self, stock_id: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
         "dataset": "TaiwanStockFinancialStatements",
         "data_id": stock_id,
         "start_date": start_date,
         "end_date": end_date
        }
       
        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "data" not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])
            return df
        except Exception as e:

            return pd.DataFrame()
        


    def get_stock_price(self, stock_id: str,
                        start_date: str, end_date: str) -> pd.DataFrame:
        """獲取股價資料（日K線）"""
        try:
            print(f"[DataFetcher] 正在獲取 {stock_id} 的股價資料...")
            df = self.loader.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            if df.empty:
                print(f"[DataFetcher] 警告：{stock_id} 股價查無資料")
            else:
                print(f"[DataFetcher] 成功獲取 {len(df)} 筆股價資料")
            return df
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取股價失敗 - {e}")
            return pd.DataFrame()

    def get_monthly_revenue(self, stock_id: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """獲取月營收資料"""
        try:
            print(f"[DataFetcher] 正在獲取 {stock_id} 的月營收資料...")
            df = self.loader.taiwan_stock_month_revenue(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            if df.empty:
                print(f"[DataFetcher] 警告：{stock_id} 月營收查無資料")
            else:
                print(f"[DataFetcher] 成功獲取 {len(df)} 筆月營收資料")
            return df
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取月營收失敗 - {e}")
            return pd.DataFrame()
        

    def get_stock_list(self):
        url = "https://api.finmindtrade.com/api/v4/data?"
        params = {
            "dataset": "TaiwanStockInfo"
        }

        response = requests.get(url, params=params)
        data = response.json()
        return data["data"]

    
    def get_per(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取本益比、殖利率、PBR"""
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # data["data"] 是 list of dict，轉成 DataFrame
        return pd.DataFrame(data["data"])
 

    def get_balance_sheet(self, stock_id: str,
                          start_date: str, end_date: str):
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockBalanceSheet",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            # 不帶 token，使用匿名額度
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if "data" not in data:
                print(f"[DataFetcher] 警告：{stock_id} 資產負債表查無資料")
                return pd.DataFrame()
                
            df = pd.DataFrame(data["data"])
            print(f"[DataFetcher] 成功獲取 {len(df)} 筆資產負債表資料")
            return df
            
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取資產負債表失敗 - {e}")
            return pd.DataFrame()

    def get_cash_flow_statement(self, stock_id: str,
                                start_date: str, end_date: str) -> pd.DataFrame:
        """獲取現金流量表資料"""
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockCashFlowsStatement",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "data" not in data:
                print(f"[DataFetcher] 警告：{stock_id} 現金流量表查無資料")
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])
            print(f"[DataFetcher] 成功獲取 {len(df)} 筆現金流量表資料")
            return df

        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取現金流量表失敗 - {e}")
            return pd.DataFrame()
        