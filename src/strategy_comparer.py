from src.data_service import DataService
from src.backtester import Backtester
from src.performance_analyzer import PerformanceAnalyzer
from tabulate import tabulate


class StrategyComparer:
    def __init__(self):
        self.data_service = DataService()
        self.backtester = Backtester()
        self.performance_analyzer = PerformanceAnalyzer()
        


    def compare_strategies(self, stock_id, start_date, end_date):
        price_df = self.data_service.get_data(stock_id, 'stock_price', start_date, end_date)
        per_df = self.data_service.get_data(stock_id, 'per', start_date, end_date)
        bh_result = self.backtester.buy_and_hold(price_df)
        pe_result = self.backtester.pe_strategy(price_df, per_df)
        ma_result = self.backtester.ma_strategy(price_df)
        rsi_result = self.backtester.rsi_strategy(price_df)
        kd_result = self.backtester.kd_strategy(price_df)

        results = []
        bah_metrics = {
            'strategy_name': 'Buy and Hold',
            'total_return': bh_result['returns'],  
            'sharpe_ratio': self.performance_analyzer.sharpe_ratio(price_df, 0)['sharpe_ratio'],
            'max_drawdown': self.performance_analyzer.max_drawdown(price_df)['max_drawdown'],
            'profit_loss_ratio': 'N/A',
            'win_rate': 'N/A'
        }
        results.append(bah_metrics)

        # PE Strategy
        pe_win_result = self.performance_analyzer.win_rate(pe_result['trades'])
        pe_metrics = {
            'strategy_name': 'PE Strategy',
            'total_return': pe_result['total_return'],   # 注意：這裡是 total_return
            'sharpe_ratio': self.performance_analyzer.sharpe_ratio(price_df, 0)['sharpe_ratio'],
            'max_drawdown': self.performance_analyzer.max_drawdown(price_df)['max_drawdown'],
            'win_rate': pe_win_result['win_rate'],        # 要計算
            'profit_loss_ratio': pe_win_result['profit_loss_ratio']
        }
        results.append(pe_metrics)

        # MA Strategy
        ma_win_result = self.performance_analyzer.win_rate(ma_result['trades'])
        ma_metrics = {
            'strategy_name': 'MA Strategy',
            'total_return': ma_result['total_return'],
            'sharpe_ratio': self.performance_analyzer.sharpe_ratio(price_df, 0)['sharpe_ratio'],
            'max_drawdown': self.performance_analyzer.max_drawdown(price_df)['max_drawdown'],
            'win_rate': ma_win_result['win_rate'],
            'profit_loss_ratio': ma_win_result['profit_loss_ratio']
        }
        results.append(ma_metrics)

        # RSI Strategy
        rsi_win_result = self.performance_analyzer.win_rate(rsi_result['trades'])
        rsi_metrics = {
            'strategy_name': 'RSI Strategy',
            'total_return': rsi_result['total_return'],
            'sharpe_ratio': self.performance_analyzer.sharpe_ratio(price_df, 0)['sharpe_ratio'],
            'max_drawdown': self.performance_analyzer.max_drawdown(price_df)['max_drawdown'],
            'win_rate': rsi_win_result['win_rate'],
            'profit_loss_ratio': rsi_win_result['profit_loss_ratio']
        }
        results.append(rsi_metrics)

        # KD Strategy
        kd_win_result = self.performance_analyzer.win_rate(kd_result['trades'])
        kd_metrics = {
            'strategy_name': 'KD Strategy',
            'total_return': kd_result['total_return'],
            'sharpe_ratio': self.performance_analyzer.sharpe_ratio(price_df, 0)['sharpe_ratio'],
            'max_drawdown': self.performance_analyzer.max_drawdown(price_df)['max_drawdown'],
            'win_rate': kd_win_result['win_rate'],
            'profit_loss_ratio': kd_win_result['profit_loss_ratio']
        }
        results.append(kd_metrics)
        self._format_table(results)
        return results


        
    def _format_table(self, results):
        headers = ['策略名稱', '總報酬(%)', 'Sharpe Ratio', '最大回撤(%)', '勝率(%)', '盈虧比']
        table_data = []
        for r in results:
            row = [
                r['strategy_name'],
                r['total_return'],
                r['sharpe_ratio'],
                r['max_drawdown'],
                r['win_rate'],
                r['profit_loss_ratio']
            ]
            table_data.append(row)

        table = tabulate(table_data, headers = headers, tablefmt = 'grid')
        print(table)



