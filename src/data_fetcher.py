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
        """獲取股價資料（日K線），並還原除權息造成的價格缺口"""
        try:
            print(f"[DataFetcher] 正在獲取 {stock_id} 的股價資料...")
            df = self.loader.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                print(f"[DataFetcher] 警告：{stock_id} 股價查無資料")
                return df

            print(f"[DataFetcher] 成功獲取 {len(df)} 筆股價資料")

            # 還原股價是「錦上添花」而非核心資料，就算除權息資料抓取/計算失敗，
            # 也不該讓整支股票的股價資料整個報廢，因此這裡用內層 try/except
            # 隔離風險：失敗時退回原始（未還原）股價，並印出警告讓使用者知道。
            try:
                dividend_df = self.get_dividend(stock_id, start_date, end_date)
                df = self._adjust_for_dividends(df, dividend_df)
            except Exception as e:
                print(f"[DataFetcher] 警告：{stock_id} 股價還原失敗，改用原始股價 - {e}")

            return df
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取股價失敗 - {e}")
            return pd.DataFrame()

    def get_dividend(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """獲取股利分派資料（現金股利、股票股利、現金增資認股）"""
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockDividend",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "data" not in data:
                return pd.DataFrame()

            return pd.DataFrame(data["data"])
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取股利資料失敗 - {e}")
            return pd.DataFrame()

    def _adjust_for_dividends(self, price_df: pd.DataFrame,
                              dividend_df: pd.DataFrame) -> pd.DataFrame:
        """
        用除權息公告，把歷史股價還原成「調整後價格」

        還原邏輯（交易所官方除權息參考價公式）：
            除權息參考價 = (前一日收盤價 − 現金股利 + 認股價 × 現金增資認股比例)
                          / (1 + 股票股利換股比例 + 現金增資認股比例)
            調整係數 = 除權息參考價 / 前一日收盤價

        還原慣例是「今天的價格不動，往回調整過去的價格」：
        某一天的調整係數 = 這天之後所有除權息事件的係數連乘。
        """
        if dividend_df.empty:
            return price_df

        df = price_df.sort_values('date').reset_index(drop=True)

        raw = dividend_df.copy()
        raw['cash_dividend'] = raw['CashEarningsDistribution'] + raw['CashStatutorySurplus']
        # 股票股利欄位是「每股配股金額」，股票面額 10 元，除以 10 換算成配股比例
        raw['stock_ratio'] = (
            raw['StockEarningsDistribution'] + raw['StockStatutorySurplus']
        ) / 10
        raw['rights_ratio'] = raw['CashIncreaseSubscriptionRate']
        raw['subscription_price'] = raw['CashIncreaseSubscriptionpRrice']

        # 除息日（CashExDividendTradingDate）跟除權日（StockExDividendTradingDate）
        # 不一定是同一天，一筆公告要依兩個日期欄位的狀態拆成 0～2 個獨立事件：
        #   ① 兩者皆有值、同一天 → 用官方統一公式合併成一筆事件
        #   ② 兩者皆有值、不同天 → 拆成「純除息」+「純除權」兩筆各自獨立的事件
        #   ③ 只有除息日有值     → 純除息事件（股票股利比例設 0）
        #   ④ 只有除權日有值     → 純除權事件（現金股利/現金增資設 0）
        #   ⑤ 兩者皆無值         → 這筆公告沒有實際發生除權息，跳過
        # 用 .iterrows() 逐列判斷再展開成不定筆數的新資料列，是因為這是「條件式地
        # 產生 0～2 筆輸出」，向量化操作不好表達這種「一列可能變兩列」的邏輯；
        # 股利公告一年最多幾筆，用迴圈處理的效能成本可忽略不計。
        event_rows = []
        for _, row in raw.iterrows():
            cash_date = row['CashExDividendTradingDate']
            stock_date = row['StockExDividendTradingDate']
            has_cash = cash_date != ''
            has_stock = stock_date != ''

            if not has_cash and not has_stock:
                continue  # 情境⑤：沒有事件

            if has_cash and has_stock and cash_date == stock_date:
                # 情境①：權息同日，統一公式一次算完
                event_rows.append({
                    'ex_date': cash_date,
                    'cash_dividend': row['cash_dividend'],
                    'stock_ratio': row['stock_ratio'],
                    'rights_ratio': row['rights_ratio'],
                    'subscription_price': row['subscription_price'],
                })
            elif has_cash and has_stock:
                # 情境②：權息分離，拆成兩筆各自獨立的事件
                event_rows.append({
                    'ex_date': cash_date,
                    'cash_dividend': row['cash_dividend'],
                    'stock_ratio': 0.0,
                    'rights_ratio': row['rights_ratio'],
                    'subscription_price': row['subscription_price'],
                })
                event_rows.append({
                    'ex_date': stock_date,
                    'cash_dividend': 0.0,
                    'stock_ratio': row['stock_ratio'],
                    'rights_ratio': 0.0,
                    'subscription_price': 0.0,
                })
            elif has_cash:
                # 情境③：純除息（現金增資認股沒有獨立日期欄位，假設跟除息日綁在一起）
                event_rows.append({
                    'ex_date': cash_date,
                    'cash_dividend': row['cash_dividend'],
                    'stock_ratio': 0.0,
                    'rights_ratio': row['rights_ratio'],
                    'subscription_price': row['subscription_price'],
                })
            else:
                # 情境④：純除權
                event_rows.append({
                    'ex_date': stock_date,
                    'cash_dividend': 0.0,
                    'stock_ratio': row['stock_ratio'],
                    'rights_ratio': 0.0,
                    'subscription_price': 0.0,
                })

        if not event_rows:
            return df

        events = pd.DataFrame(event_rows).sort_values('ex_date').reset_index(drop=True)

        # merge_asof 要求 key 欄位是數值或 datetime，不能是字串（object dtype），
        # 所以另外開暫存的 datetime 欄位做比對，不動原本的 'date' 字串欄位，
        # 避免影響其他程式碼對日期格式（ISO 字串）的既有假設。
        df['_date_dt'] = pd.to_datetime(df['date'])
        events['_ex_date_dt'] = pd.to_datetime(events['ex_date'])

        # merge_asof 是「模糊 join」：對 events 的每個 ex_date，往「過去」方向
        # 找股價表裡最近的一筆日期，用它的收盤價當作公式裡的「前一日收盤價」
        # （direction='backward' 是預設值，概念上類似 C++ std::upper_bound 後再往前一格）
        # allow_exact_matches=False：merge_asof 預設「剛好等於」也算命中，但除息日
        # 當天不能拿自己的收盤價當「前一日收盤價」，所以強制要求嚴格早於 ex_date。
        events = pd.merge_asof(
            events,
            df[['_date_dt', 'close']].rename(columns={'close': 'prev_close'}),
            left_on='_ex_date_dt', right_on='_date_dt',
            direction='backward', allow_exact_matches=False
        )

        denominator = 1 + events['stock_ratio'] + events['rights_ratio']
        numerator = (
            events['prev_close'] - events['cash_dividend']
            + events['rights_ratio'] * events['subscription_price']
        )
        events['factor'] = (numerator / denominator) / events['prev_close']

        # 反向累積乘積：[::-1] 是 Python 的 slice 語法糖，把序列倒過來讀取
        # （不複製資料、只是換個方向），cumprod() 再算累積乘積，兩次 [::-1]
        # 等於「從未來往過去累乘」。這是後綴乘積（suffix product），
        # 跟一般常見由左到右的前綴和（prefix sum，C++ std::partial_sum）方向相反。
        events['cum_factor'] = events['factor'][::-1].cumprod()[::-1]

        # 這次 merge_asof 用 direction='forward'：對每一筆股價日期，
        # 往「未來」方向找最近的一筆除權息事件，把它的累積係數帶回來。
        # 同樣要 allow_exact_matches=False：除息日當天的收盤價本身就是「新」的
        # 參考價，不該再被自己這筆事件的係數往回調整一次。
        df = pd.merge_asof(
            df,
            events[['_ex_date_dt', 'cum_factor']].rename(columns={'_ex_date_dt': '_date_dt'}),
            on='_date_dt', direction='forward', allow_exact_matches=False
        )
        # 找不到未來事件的日期（例如最新的交易日之後已無除權息），
        # 代表不需要調整，用 1.0 補起來（乘上 1 等於維持原值）
        df['cum_factor'] = df['cum_factor'].fillna(1.0)

        for col in ['open', 'max', 'min', 'close']:
            if col in df.columns:
                df[col] = df[col] * df['cum_factor']

        return df.drop(columns=['cum_factor', '_date_dt'])

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

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if "data" not in data:
                print(f"[DataFetcher] 錯誤：股票清單查無資料 - {data.get('msg', '未知錯誤')}")
                return []

            return data["data"]
        except Exception as e:
            print(f"[DataFetcher] 錯誤：獲取股票清單失敗 - {e}")
            return []

    
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
        

