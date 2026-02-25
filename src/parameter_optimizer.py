from itertools import product
from src.data_service import DataService
from src.backtester import Backtester
from tabulate import tabulate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class ParameterOptimizer :
    PARAM_GRIDS = {
    'ma': {
        'short_window': [3, 5, 10, 15, 20],
        'long_window': [20, 30, 50, 60]
    },
    'rsi': {
        'period': [7, 10, 14, 20],
        'buy_rsi': [20, 25, 30, 35],
        'sell_rsi': [65, 70, 75, 80]
    },
    'kd': {
        'period': [5, 7, 9, 12, 14]
    },
    'pe': {
        'buy_pe': [10, 12, 15, 18],
        'sell_pe': [20, 25, 30, 35]
    }
}



    def __init__(self):
        self.data_service = DataService()
        self.back_tester = Backtester()


    def optimize(self, stock_id, start_date, end_date, strategy_name):
        price_df = self.data_service.get_data(stock_id, 'stock_price', start_date, end_date)

        if strategy_name == 'pe':
            per_df = self.data_service.get_data(stock_id, 'per', start_date, end_date)
        
        strategy_grid = self._generate_param_grid(strategy_name)
        keys = list(strategy_grid.keys())
        values = list(strategy_grid.values())
        results = []
        strategy_methods = {
                    'ma': self.back_tester.ma_strategy,
                    'rsi': self.back_tester.rsi_strategy,
                    'kd': self.back_tester.kd_strategy,
                    'pe': self.back_tester.pe_strategy
                }
        
        for combo in product(*values):
            params = dict(zip(keys, combo))
            
            if strategy_name == 'pe':
                result = self.back_tester.pe_strategy(price_df, per_df, **params)   
            else:
                result = strategy_methods[strategy_name](price_df, **params)

            results.append({
            **params,                              
            'total_return': result['total_return']
        })

        self._format_result(results, strategy_name)
        return results
        



    def _generate_param_grid(self, strategy_name):
        return self.PARAM_GRIDS[strategy_name]

    def _format_result(self, results, strategy_name):
        results.sort(key=lambda x : x['total_return'], reverse=True)

        print(tabulate(results, headers='keys', tablefmt='grid'))

        best = results[0]
        print(f'\n最佳參數: {best}')

        top5 = [r['total_return'] for r in results[:5]]
        spread = max(top5) - min(top5)
        avg = sum(top5) / len(top5)
        stability = spread / avg * 100

        print(f'穩健性 : {stability:.2f}%')
        if stability < 20:
            print('穩健性：高（參數高原）')
        elif stability < 50:
            print('穩健性：中')
        else:
            print('穩健性：低（孤立山峰，小心過擬合）')

        df = pd.DataFrame(results)
       

        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False  

        keys = [k for k in results[0].keys() if k != 'total_return']

        if len(keys) == 2:
            pivot = df.pivot_table(index=keys[1], columns=keys[0], values='total_return')
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn')
            plt.title(f'{strategy_name.upper()} 參數優化 Heatmap')
            plt.show()
        elif len(keys) == 1:
            # KD 只有一個參數，用 bar chart
            plt.bar(df[keys[0]].astype(str), df['total_return'])
            plt.title(f'{strategy_name.upper()} 參數優化')
            plt.ylabel('報酬率 (%)')
            plt.show()

        elif len(keys) >= 3:
            fixed_key = keys[0]
            for val in df[fixed_key].unique():
                sub_df = df[df[fixed_key] == val]
                pivot = sub_df.pivot_table(index=keys[2], columns=keys[1], values='total_return')
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn')
                plt.title(f'{strategy_name.upper()} ({fixed_key}={val})')
                plt.show()
    

        self._plateau_best(results, strategy_name)




    def _plateau_best(self, results, strategy_name):
        grid = self.PARAM_GRIDS[strategy_name]  # grid為多個dict
        keys = list(grid.keys()) # keys(list)為grid的key

        best_score = -float('inf') 
        best_params = None  

        for r in results: # results為多個dict的list
            neighbor_returns = [] 
            for other in results:
                is_neighbor = True
                for key in keys:
                    idx_r = grid[key].index(r[key])
                    idx_o = grid[key].index(other[key])
                    if abs(idx_r - idx_o) > 1: # 抓九宮格
                        is_neighbor = False
                        break
                if is_neighbor:
                    neighbor_returns.append(other['total_return'])

            avg = sum(neighbor_returns) / len(neighbor_returns)
            if avg > best_score:
                best_score = avg
                best_params = r

        print(f'\n高原最佳參數: {best_params}')
        print(f'鄰居平均報酬: {best_score:.2f}%')


