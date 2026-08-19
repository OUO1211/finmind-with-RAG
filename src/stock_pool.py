"""
StockPool：股票池定錨
功能：以每日市值前 N 大動態定義可交易股票池，避免用固定期間排名造成 look-ahead bias
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .data_service import DataService

# MI_INDEX 回傳「全部有價證券」（含權證、ETF、存託憑證等），
# 這些類別不是一般上市公司股票，算市值排名時要排除
_NON_COMMON_STOCK_CATEGORIES = {'ETF', 'ETN', 'Index', '存託憑證', '受益證券', '大盤'}


class StockPool:
    """
    股票池定錨

    職責：
    1. 抓證交所每日全市場收盤價（平行 + 節流）
    2. 用 FinMind 股本資料換算市值
    3. 逐日排出市值前 N 大，只影響「進場判斷」——已持有部位不因排名變動被強制平倉
       （出場邏輯交給 risk_manager 既有的停損/停利與策略訊號判斷）
    """

    TOP_N = 50
    MAX_WORKERS = 3           # 保守併發數，避免打爆證交所/FinMind
    REQUEST_INTERVAL = 0.5    # 送出下一個請求前的節流間隔（秒）

    # requests 預設的 User-Agent 會直接暴露「這是程式化存取」（例如
    # "python-requests/2.x"），部分網站（尤其像證交所這種舊式後端）的
    # 防爬蟲機制會直接擋掉這類請求。帶一個瀏覽器等級的 User-Agent，
    # 讓請求看起來像一般使用者從瀏覽器發出，降低被擋的機率。
    _BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    def __init__(self, cache_dir: str = "data/stock_pool", token: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_service = DataService(token=token)

        # 用同一個 Session 重複使用 TCP 連線，避免每次請求都重新建立連線
        # （對伺服器來說，短時間內大量「新連線」本身也是常見的機器人特徵）
        self._twse_session = requests.Session()
        self._twse_session.headers.update(self._BROWSER_HEADERS)

    def get_trading_days(self, reference_stock_id: str,
                         start_date: str, end_date: str) -> list:
        """用一檔已知股票的股價資料，反推回測期間有哪些真正的交易日"""
        price_df = self.data_service.get_data(
            reference_stock_id, 'stock_price', start_date, end_date
        )
        return sorted(price_df['date'].unique().tolist())

    STOCK_LIST_CACHE_DAYS = 7  # 股票清單變動不頻繁（新股上市/下市才會變），快取久一點

    def _common_stock_ids(self) -> set:
        """
        從 FinMind 股票清單篩出「一般上市普通股」的代號集合

        本地快取這份清單，不是每次 build() 都重打 get_stock_list()：這支 API
        實測會遇到不定期的暫時性封鎖，而股票清單本身變動很少，沒必要讓整個
        股票池建構流程每次都依賴這一次呼叫是否剛好成功。
        """
        cache_file = self.cache_dir / "common_stock_ids.csv"

        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - mtime).days <= self.STOCK_LIST_CACHE_DAYS:
                print(f"[StockPool] 命中股票清單快取：{cache_file}")
                return set(pd.read_csv(cache_file)['stock_id'].astype(str))

        stocks = self.data_service.fetcher.get_stock_list()
        if not stocks:
            # 抓取失敗時，寧可用本地過期的舊快取，也比完全沒有股票清單好；
            # 只有在「連過期快取都沒有」時才真的回傳空集合。
            if cache_file.exists():
                print(f"[StockPool] 警告：股票清單抓取失敗，改用過期快取：{cache_file}")
                return set(pd.read_csv(cache_file)['stock_id'].astype(str))
            print("[StockPool] 警告：股票清單抓取失敗，且無任何快取可用")
            return set()

        common_ids = {
            s['stock_id'] for s in stocks
            if s.get('type') == 'twse'
            and s.get('industry_category') not in _NON_COMMON_STOCK_CATEGORIES
        }

        pd.DataFrame({'stock_id': sorted(common_ids)}).to_csv(cache_file, index=False)
        print(f"[StockPool] 股票清單已存入快取：{cache_file}")

        return common_ids

    def _fetch_market_snapshot(self, date: str, max_retries: int = 3) -> pd.DataFrame:
        """
        打證交所 MI_INDEX，抓某一天全市場的收盤價

        帶重試機制：證交所這個舊版端點沒有公開的速率限制文件，實測發現
        短時間內請求量過大會被暫時性封鎖（回應變成空字串，JSON 解析失敗）。
        遇到這種狀況用指數退避（exponential backoff）重試：等待時間隨每次
        重試倍增（2 秒 → 4 秒 → 8 秒），給伺服器端的暫時性狀態足夠時間恢復。
        """
        date_str = date.replace('-', '')
        url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
        params = {"date": date_str, "type": "ALL", "response": "json"}

        for attempt in range(max_retries):
            try:
                response = self._twse_session.get(url, params=params, timeout=15)
                payload = response.json()

                if payload.get('stat') != 'OK':
                    return pd.DataFrame()

                # tables[8] 是「每日收盤行情（個股）」表；其他 index 是大盤統計等
                # 其他表格。這是憑實際測試回應結構確認的固定位置，不是文件記載的
                # 穩定 API 承諾，若證交所改版格式，這裡要重新確認。
                table = payload['tables'][8]
                rows = table['data']
                if not rows:
                    return pd.DataFrame()

                df = pd.DataFrame(rows, columns=[
                    'stock_id', 'name', 'volume', 'transactions', 'value',
                    'open', 'high', 'low', 'close', 'change_sign', 'change',
                    'best_bid', 'best_bid_volume', 'best_ask', 'best_ask_volume', 'pe_ratio'
                ])
                df['date'] = date

                # 數值欄位帶千分位逗號（例如 "11,515,134"），要先去逗號再轉數字；
                # 當天沒成交的證券欄位會是 "--"，轉換失敗的值用 errors='coerce'
                # 轉成 NaN，之後會被 dropna 篩掉，不會讓整批資料因個別壞值報錯中斷。
                df['close'] = pd.to_numeric(
                    df['close'].astype(str).str.replace(',', '', regex=False),
                    errors='coerce'
                )

                return df[['date', 'stock_id', 'close']]
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    time.sleep(backoff)
                else:
                    print(f"[StockPool] 警告：{date} 全市場快照抓取失敗（已重試 {max_retries} 次）- {e}")

        return pd.DataFrame()

    def _fetch_all_snapshots(self, trading_days: list) -> pd.DataFrame:
        """
        平行抓取多個交易日的全市場快照，分批處理並偵測失敗率

        不是把所有交易日一次全部丟進 thread pool，而是分成小批次依序處理，
        每批之間可以檢查「這批失敗率高不高」——如果高，代表可能被暫時性
        封鎖了，主動冷卻(sleep 更久)再繼續，而不是繼續用同樣的節奏硬撞。
        """
        BATCH_SIZE = 20
        COOLDOWN_SECONDS = 60
        FAILURE_RATE_THRESHOLD = 0.5

        frames = []
        batches = [
            trading_days[i:i + BATCH_SIZE]
            for i in range(0, len(trading_days), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                futures = {}
                for date in batch:
                    futures[executor.submit(self._fetch_market_snapshot, date)] = date
                    time.sleep(self.REQUEST_INTERVAL)

                batch_results = {}
                for future in as_completed(futures):
                    date = futures[future]
                    batch_results[date] = future.result()

            batch_frames = [df for df in batch_results.values() if not df.empty]
            failure_count = len(batch) - len(batch_frames)
            failure_rate = failure_count / len(batch)

            frames.extend(batch_frames)
            print(f"[StockPool] 第 {batch_idx + 1}/{len(batches)} 批完成，"
                  f"成功 {len(batch_frames)}/{len(batch)}")

            # 失敗率過高，判斷可能觸發了暫時性封鎖，先冷卻一段時間再繼續下一批，
            # 避免整個剩餘流程都在對著已經被擋的連線持續失敗。
            is_last_batch = batch_idx == len(batches) - 1
            if failure_rate > FAILURE_RATE_THRESHOLD and not is_last_batch:
                print(f"[StockPool] 失敗率 {failure_rate:.0%} 過高，"
                      f"冷卻 {COOLDOWN_SECONDS} 秒後繼續...")
                time.sleep(COOLDOWN_SECONDS)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _get_capital_stock(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """取得單一股票的股本歷史（date 已經過時間軸防污染處理，是生效日期不是季度結束日）"""
        try:
            bs = self.data_service.get_data(stock_id, 'balance_sheet', start_date, end_date)
        except Exception as e:
            print(f"[StockPool] 警告：{stock_id} 股本資料抓取失敗 - {e}")
            return pd.DataFrame()

        if bs.empty or 'type' not in bs.columns:
            return pd.DataFrame()

        capital = bs[bs['type'] == 'CapitalStock'][['date', 'value']].copy()
        if capital.empty:
            return pd.DataFrame()

        capital = capital.rename(columns={'value': 'capital_stock'})
        capital['stock_id'] = stock_id
        return capital.sort_values('date').reset_index(drop=True)

    def _fetch_all_capital_stock(self, stock_ids: list,
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """平行抓取多檔股票的股本歷史"""
        frames = []

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}
            for stock_id in stock_ids:
                futures[executor.submit(
                    self._get_capital_stock, stock_id, start_date, end_date
                )] = stock_id
                time.sleep(self.REQUEST_INTERVAL)

            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def build(self, reference_stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        建立回測期間內，每個交易日市值前 N 大的股票池

        Args:
            reference_stock_id: 用來反推交易日曆的參考股票（例如 '2330'）
            start_date, end_date: 回測期間

        Returns:
            DataFrame，欄位為 ['date', 'stock_id']，代表這一天這檔股票是市值前 N 大
        """
        trading_days = self.get_trading_days(reference_stock_id, start_date, end_date)
        print(f"[StockPool] 共 {len(trading_days)} 個交易日，開始抓取全市場快照...")

        price_df = self._fetch_all_snapshots(trading_days)
        if price_df.empty:
            print("[StockPool] 警告：全市場快照抓取失敗，回傳空股票池")
            return pd.DataFrame(columns=['date', 'stock_id'])

        common_ids = self._common_stock_ids()
        price_df = price_df[price_df['stock_id'].isin(common_ids)].copy()
        price_df = price_df.dropna(subset=['close'])

        candidate_ids = sorted(price_df['stock_id'].unique().tolist())
        print(f"[StockPool] 篩選出 {len(candidate_ids)} 檔一般股票，開始抓取股本資料...")

        capital_df = self._fetch_all_capital_stock(candidate_ids, start_date, end_date)
        if capital_df.empty:
            print("[StockPool] 警告：股本資料抓取失敗，回傳空股票池")
            return pd.DataFrame(columns=['date', 'stock_id'])

        price_df['_date_dt'] = pd.to_datetime(price_df['date'])
        capital_df['_date_dt'] = pd.to_datetime(capital_df['date'])

        # merge_asof 搭配 by='stock_id'：逐股票各自對每個交易日往「過去」方向
        # 找最近一次已公告的股本（股本是季度資料，交易日之間要「延續」最近一次
        # 的值，不是只有公告當天才有數字）。兩邊都要先依 '_date_dt' 排序。
        merged = pd.merge_asof(
            price_df.sort_values('_date_dt'),
            capital_df.sort_values('_date_dt')[['_date_dt', 'stock_id', 'capital_stock']],
            on='_date_dt', by='stock_id', direction='backward'
        )
        merged = merged.dropna(subset=['capital_stock', 'close'])

        # 股票面額固定 10 元，股本金額 / 10 = 在外流通股數
        merged['market_cap'] = merged['close'] * (merged['capital_stock'] / 10)

        # groupby('date') 把資料依交易日分組，每組各自排序取前 N 筆——
        # 概念上等同對每個 key 分開跑一次「排序 + 取前 k 筆」，pandas 會
        # 自動逐組處理，不用自己寫迴圈跑 386 個交易日。
        pool = (
            merged.sort_values('market_cap', ascending=False)
            .groupby('date')
            .head(self.TOP_N)
            [['date', 'stock_id']]
            .sort_values(['date', 'stock_id'])
            .reset_index(drop=True)
        )

        cache_file = self.cache_dir / f"pool_{start_date}_{end_date}.csv"
        pool.to_csv(cache_file, index=False)
        print(f"[StockPool] 股票池已建立並存入快取：{cache_file}")

        return pool

    @staticmethod
    def to_lookup(pool_df: pd.DataFrame) -> dict:
        """
        把股票池表格轉成執行期查詢用的 dict of set

        對應 C++ 的 std::unordered_map<std::string, std::unordered_set<std::string>>，
        回測迴圈裡用 `stock_id in lookup[date]` 做 O(1) 平均時間複雜度的成員查詢，
        比在迴圈裡重複對 DataFrame 做過濾快得多。
        """
        lookup = {}
        for date, group in pool_df.groupby('date'):
            lookup[date] = set(group['stock_id'])
        return lookup
