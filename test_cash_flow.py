import pandas as pd
from src.cash_flow_analyzer import CashFlowAnalyzer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cfa = CashFlowAnalyzer()

print('=== 現金流量摘要 ===')
summary = cfa.get_summary('2330', '2020-01-01', '2024-12-31')
print(summary)

print('\n=== 自由現金流 ===')
fcf = cfa.free_cash_flow('2330', '2020-01-01', '2024-12-31')
print(fcf)

cfa.plot_summary('2330', '2020-01-01', '2024-12-31')
cfa.plot_fcf('2330', '2020-01-01', '2024-12-31')