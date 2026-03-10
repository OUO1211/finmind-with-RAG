from src.backtester import Backtester
from src.data_service import DataService
from src.risk_manager import RiskManager

ds = DataService()
bt = Backtester()
rm = RiskManager()

price_df = ds.get_data('2330', 'stock_price', '2024-01-01', '2024-12-31')

# 原始 MA 策略
original = bt.ma_strategy(price_df)
print('=== 原始 ===')
print(f'總報酬: {original["total_return"]}%')
print(f'交易次數: {original["total_trades"]}')

# 加停損/停利
adjusted = rm.apply_stops(price_df, original, stop_loss=-10, take_profit=20)
print('\n=== 停損-10% / 停利+20% ===')
print(f'總報酬: {adjusted["total_return"]}%')
print(f'交易次數: {adjusted["total_trades"]}')

print('\n=== 原始交易明細 ===')
for t in original['trades']:
    print(t)

print('\n=== 停損停利交易明細 ===')
for t in adjusted['trades']:
    print(t)

print('\n=== Kelly Criterion ===')
ratio = rm.kelly_criterion(original['trades'])


print('\n=== RSI 查詢 ===')
stock_id = input('請輸入股票代碼：').strip()
start_date = input('請輸入開始日期（例如 2024-01-01）：').strip()
end_date = input('請輸入結束日期（例如 2024-12-31）：').strip()
period = input('請輸入 RSI 週期（預設 14，按 Enter 跳過）：').strip()
period = int(period) if period else 14
rsi = rm.get_rsi(stock_id, start_date, end_date, period)


print('\n=== Heat Check ===')
result = rm.heat_check(stock_id, start_date, end_date)

print('\n=== Z-Score ===')
rm.z_score(stock_id, start_date, end_date)



