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

# 測試 MA 策略
print("\n===== MA 均線策略回測（5日/20日）=====")
ma_result = bt.ma_strategy(price_df, short_window=5, long_window=20)

print(f"總交易次數：{ma_result['total_trades']}")
print(f"總報酬率：{ma_result['total_return']}%")

if ma_result['trades']:
    print("\n各筆交易明細：")
    for i, trade in enumerate(ma_result['trades'], 1):
        print(f"  交易 {i}: 買入 {trade['buy_date']} @ {trade['buy_price']}, "
              f"賣出 {trade['sell_date']} @ {trade['sell_price']}, "
              f"報酬 {trade['return']}%")

# 測試 RSI 策略
print("\n===== RSI 策略回測（RSI < 30 買 / RSI > 70 賣）=====")
rsi_result = bt.rsi_strategy(price_df, period=14, buy_rsi=30, sell_rsi=70)

print(f"總交易次數：{rsi_result['total_trades']}")
print(f"總報酬率：{rsi_result['total_return']}%")

if rsi_result['trades']:
    print("\n各筆交易明細：")
    for i, trade in enumerate(rsi_result['trades'], 1):
        print(f"  交易 {i}: 買入 {trade['buy_date']} @ {trade['buy_price']}, "
              f"賣出 {trade['sell_date']} @ {trade['sell_price']}, "
              f"報酬 {trade['return']}%")

# 測試 KD 策略
print("\n===== KD 策略回測（黃金交叉買 / 死亡交叉賣）=====")
kd_result = bt.kd_strategy(price_df, period=9)

print(f"總交易次數：{kd_result['total_trades']}")
print(f"總報酬率：{kd_result['total_return']}%")

if kd_result['trades']:
    print("\n各筆交易明細：")
    for i, trade in enumerate(kd_result['trades'], 1):
        print(f"  交易 {i}: 買入 {trade['buy_date']} @ {trade['buy_price']}, "
              f"賣出 {trade['sell_date']} @ {trade['sell_price']}, "
              f"報酬 {trade['return']}%")



# 測試最大回撤
from src.performance_analyzer import PerformanceAnalyzer
pa = PerformanceAnalyzer()

print("\n===== 最大回撤分析 =====")
dd_result = pa.max_drawdown(price_df)
print(f"最大回撤：{dd_result['max_drawdown']}%")
print(f"高點：{dd_result['peak_date']} @ {dd_result['peak_price']}")
print(f"低點：{dd_result['through_date']} @ {dd_result['through_price']}")



print("\n===== Sharpe Ratio 分析 =====")
sharpe_result = pa.sharpe_ratio(price_df)
print(f"Sharpe Ratio：{sharpe_result['sharpe_ratio']}")
print(f"平均日報酬：{sharpe_result['avg_daily_return']}%")
print(f"日波動度：{sharpe_result['daily_volatility']}%")


