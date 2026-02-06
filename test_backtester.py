"""
測試 Backtester 回測引擎
"""

from src.data_service import DataService
from src.backtester import Backtester

# 初始化
ds = DataService()
bt = Backtester()

# 取得資料
print("正在獲取資料...")
price_df = ds.get_data('2330', 'stock_price', '2024-01-01', '2025-12-31')
per_df = ds.get_data('2330', 'per', '2024-01-01', '2025-12-31')

print(f"股價資料筆數：{len(price_df)}")
print(f"PE 資料筆數：{len(per_df)}")

# 測試 buy_and_hold
print("\n===== Buy and Hold 回測 =====")
bh_result = bt.buy_and_hold(price_df)
print(f"買入日期：{bh_result['buy_date']}")
print(f"賣出日期：{bh_result['sell_date']}")
print(f"買入價：{bh_result['buy_price']}")
print(f"賣出價：{bh_result['sell_price']}")
print(f"報酬率：{bh_result['returns']}%")

# 測試 PE 策略
print("\n===== PE 策略回測 =====")
result = bt.pe_strategy(price_df, per_df, buy_pe=20, sell_pe=30)

print(f"總交易次數：{result['total_trades']}")
print(f"總報酬率：{result['total_return']}%")

if result['trades']:
    print("\n各筆交易明細：")
    for i, trade in enumerate(result['trades'], 1):
        print(f"  交易 {i}: 買入 {trade['buy_date']} @ {trade['buy_price']}, "
              f"賣出 {trade['sell_date']} @ {trade['sell_price']}, "
              f"報酬 {trade['return']}%")
