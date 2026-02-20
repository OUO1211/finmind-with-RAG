from src.strategy_comparer import StrategyComparer

# 建立比較器
sc = StrategyComparer()

# 執行比較（台積電 2024 年）
print("===== 台積電 2024 策略比較 =====\n")
results = sc.compare_strategies('2330', '2024-01-01', '2024-12-31')

print(f"\n共比較了 {len(results)} 個策略")

