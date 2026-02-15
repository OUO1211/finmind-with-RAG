"""
PerformanceAnalyzer：績效分析工具
功能：計算回測策略的風險與績效指標
"""

import pandas as pd


class PerformanceAnalyzer:

    def max_drawdown(self, price_df: pd.DataFrame) -> dict:
        df = price_df[['date', 'close']].sort_values('date').reset_index(drop=True)
        df['peak'] = df['close'].cummax()
        df['drawdown'] = (df['peak'] - df['close']) / df['peak'] * 100
        max_dd_idx = df['drawdown'].idxmax()
        peak_price = df.loc[max_dd_idx, 'peak']

        return{
            'max_drawdown': round(df.loc[max_dd_idx, 'drawdown'], 2),
            'peak_date': df[df['close'] == peak_price].iloc[0]['date'],
            'peak_price':  df[df['close'] == peak_price].iloc[0]['close'],
            'through_date': df.loc[max_dd_idx, 'date'],
            'through_price': df.loc[max_dd_idx, 'close']
        }
    

    def sharpe_ratio(self, price_df: pd.DataFrame, risk_free_rate = 0) -> dict:
        df = price_df
        df['daily_return'] = df['close'].pct_change()

        avg_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()

        # 一年約 252 個交易日
        daily_rf = risk_free_rate / 252
        sharpe = (avg_return - daily_rf) / std_return * (252 ** 0.5)

        return {
        'sharpe_ratio': round(sharpe, 2),
        'avg_daily_return': round(avg_return * 100, 4),
        'daily_volatility': round(std_return * 100, 4),
        }



    def win_rate(self, trades: list) -> dict:
        wins = [t for t in trades if t['return'] > 0]
        losses = [t for t in trades if t['return'] <= 0]

        if len(trades) == 0:
            return {'win_rate': 0,
                    'avg_profit': 0, 
                    'avg_loss': 0, 
                    'profit_loss_ratio': 0}
        
        if len(wins) == 0:
            return {'win_rate': 0,
                    'avg_profit': 0, 
                    'avg_loss': round(sum(t['return'] for t in losses) / len(losses), 2),
                    'profit_loss_ratio': 0}
        
        if len(losses) == 0:
            return {'win_rate': 100,
                    'avg_profit': sum(t['return'] for t in wins) / len(wins),
                    'avg_loss': 0, 
                    'profit_loss_ratio': 0}  

        
        win_rate = len(wins) / len(trades) * 100
        avg_profit = sum(t['return'] for t in wins) / len(wins)
        avg_loss = sum(t['return'] for t in losses) / len(losses)
        profit_loss_ratio = avg_profit / abs(avg_loss)

        

        return {
            'win_rate': round(win_rate, 2),
            'avg_profit': round(avg_profit, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_loss_ratio': round(profit_loss_ratio, 2)
        }


