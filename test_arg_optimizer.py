from src.parameter_optimizer import ParameterOptimizer

po = ParameterOptimizer()

stock_id = input('請輸入股票代碼：').strip()
start_date = input('請輸入開始日期（例如 2024-01-01）：').strip()
end_date = input('請輸入結束日期（例如 2024-12-31）：').strip()
strategy_name = input('請輸入策略名稱（ma/rsi/kd/pe）：').strip()

po.optimize(stock_id, start_date, end_date, strategy_name)