from src.data_service import DataService
from src.backtester import Backtester







class RiskManager:
    def __init__(self):
        self.data_service = DataService()
        self.backtester = Backtester()


    def apply_stops(self, price_df, original_result, stop_loss, take_profit):
        adjusted_trade = []

        for trade in original_result['trades']:
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']

            holding_period = price_df[
            (price_df['date'] >= buy_date) & (price_df['date'] <= sell_date)
        ]
            
            triggered = False
            for _, row in holding_period.iterrows():
                change = (row['close'] - buy_price) / buy_price * 100
                if(change > take_profit or change < stop_loss):
                    new_trade = {
                        'buy_date': buy_date,
                        'sell_date': row['date'],
                        'buy_price': buy_price,
                        'sell_price': row['close'],
                        'return': round(change, 2)
                    }
                    adjusted_trade.append(new_trade)
                    triggered = True
                    break
                    
            if not triggered:
                adjusted_trade.append(trade)

        total_return = sum(t['return'] for t in adjusted_trade)
        return {
            'trades': adjusted_trade,
            'total_trades': len(adjusted_trade),
            'total_return': round(total_return, 2)
        }
    
    def kelly_criterion(self, trades):
                from src.performance_analyzer import PerformanceAnalyzer
                pa = PerformanceAnalyzer()
                stats = pa.win_rate(trades)

                p = stats['win_rate'] / 100
                b = stats['profit_loss_ratio']
                q = 1 - p

                if b == 0:
                    return 0

                kelly = (b * p - q) / b
                half_kelly = kelly / 2

                print(f'勝率: {p*100:.1f}%')
                print(f'盈虧比: {b:.2f}')
                print(f'Full Kelly: {kelly*100:.1f}%')
                print(f'Half Kelly: {half_kelly*100:.1f}%')

                return round(half_kelly, 4)
        
    

        

    


        
        
