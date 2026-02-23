from src.strategy_comparer import StrategyComparer

# 建立比較器
sc = StrategyComparer()
stock_ids = input('請輸入股票代碼（逗號分隔）：').split(',')
start_date = input('請輸入開始日期（例如 2024-01-01）：')
end_date = input('請輸入結束日期（例如 2024-12-31）：')

stock_ids = [s.strip() for s in stock_ids]


results = sc.compare_strategies(stock_ids, start_date, end_date)

print(f"\n共比較了 {len(results)} 個策略")

