from src.data_service import DataService
from src.backtester import Backtester
from src.performance_analyzer import PerformanceAnalyzer

ds = DataService()
bt = Backtester()
pa = PerformanceAnalyzer()

price_df = ds.get_data('2330', 'stock_price', '2024-01-01', '2025-12-31')

ma_result = bt.ma_strategy(price_df, short_window=5, long_window=20)
rsi_result = bt.rsi_strategy(price_df, period=14, buy_rsi=30, sell_rsi=70)
kd_result = bt.kd_strategy(price_df, period=9)

print("\n===== 勝率分析（MA 策略）=====")
wr_result = pa.win_rate(ma_result['trades'])
print(f"勝率：{wr_result['win_rate']}%")
print(f"平均獲利：{wr_result['avg_profit']}%")
print(f"平均虧損：{wr_result['avg_loss']}%")
print(f"盈虧比：{wr_result['profit_loss_ratio']}")

print("\n===== 勝率分析（RSI 策略）=====")
wr_result = pa.win_rate(rsi_result['trades'])
print(f"勝率：{wr_result['win_rate']}%")
print(f"平均獲利：{wr_result['avg_profit']}%")
print(f"平均虧損：{wr_result['avg_loss']}%")
print(f"盈虧比：{wr_result['profit_loss_ratio']}")

print("\n===== 勝率分析（KD 策略）=====")
wr_result = pa.win_rate(kd_result['trades'])
print(f"勝率：{wr_result['win_rate']}%")
print(f"平均獲利：{wr_result['avg_profit']}%")
print(f"平均虧損：{wr_result['avg_loss']}%")
print(f"盈虧比：{wr_result['profit_loss_ratio']}")