from src.data_service import DataService
from src.backtester import Backtester
from src.performance_analyzer import PerformanceAnalyzer

ds = DataService()
bt = Backtester()
pa = PerformanceAnalyzer()

print('Enter start_date:')
start_date = input()
print('Enter end_date:')
end_date = input()
print('Enter stock_id:')
stock_id = input()

price_df = ds.get_data(stock_id, 'stock_price', start_date, end_date)
result = bt.ma_strategy(price_df)
pa.plot_performance(result['equity_curve'], 'MA Strategy')

# RSI
result = bt.rsi_strategy(price_df)
pa.plot_performance(result['equity_curve'], 'RSI Strategy')

# KD
result = bt.kd_strategy(price_df)
pa.plot_performance(result['equity_curve'], 'KD Strategy')

# Buy and Hold
result = bt.buy_and_hold(price_df)
pa.plot_performance(result['equity_curve'], 'Buy and Hold')

