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
    

    def get_rsi(self, stock_id, start_date, end_date, period=14):
        price_df = self.data_service.get_data(stock_id, 'stock_price', start_date, end_date)
        df = price_df.sort_values('date').reset_index(drop=True)
        
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - 100 / (1 + rs)
        
        latest_rsi = df['RSI'].iloc[-1]
        print(f'{stock_id} 最新 RSI({period}): {latest_rsi:.2f}')
        
        if latest_rsi > 70:
            print('狀態: 超買（可能過熱）')
        elif latest_rsi < 30:
            print('狀態: 超賣（可能被低估）')
        else:
            print('狀態: 正常')
        
        return round(latest_rsi, 2)    
    


    def heat_check(self, stock_id, start_date, end_date):
        price_df = self.data_service.get_data(stock_id, 'stock_price', start_date, end_date)
        df = price_df.sort_values('date').reset_index(drop=True)

        # RSI(14)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        latest_rsi = (100 - 100 / (1 + rs)).iloc[-1]

        # MA60 位置
        latest_price = df['close'].iloc[-1]
        latest_ma60 = df['close'].rolling(60).mean().iloc[-1]
        above_ma = latest_price > latest_ma60

        # 乖離率
        bias = (latest_price - latest_ma60) / latest_ma60 * 100

        # 近期回撤（從 60 日高點算）
        recent_high = df['close'].rolling(60).max().iloc[-1]
        latest_drawdown = (latest_price - recent_high) / recent_high * 100

        # 輸出
        print(f'=== {stock_id} Heat Check ===')
        print(f'RSI(14): {latest_rsi:.2f}')
        print(f'股價: {latest_price:.2f} / MA60: {latest_ma60:.2f} ({"上方" if above_ma else "下方"})')
        print(f'乖離率: {bias:.1f}%')
        print(f'近期回撤: {latest_drawdown:.2f}%')

        # 綜合判斷
        signals = []

        if latest_rsi > 70:
            signals.append('RSI 超買')
        elif latest_rsi < 30:
            signals.append('RSI 超賣')

        if not above_ma:
            signals.append('跌破 MA60')

        if bias > 30:
            signals.append(f'乖離率過高 ({bias:.1f}%)')

        if latest_drawdown < -20:
            signals.append('空頭（回撤 > 20%）')
        elif latest_drawdown < -10:
            signals.append('修正（回撤 > 10%）')
        elif latest_drawdown < -5:
            signals.append('小回檔')

        if not signals:
            print('判斷: 正常')
        else:
            print(f'判斷: {", ".join(signals)}')

        return {
            'rsi': round(latest_rsi, 2),
            'price': latest_price,
            'ma60': round(latest_ma60, 2),
            'above_ma': above_ma,
            'drawdown': round(latest_drawdown, 2),
            'bias': round(bias, 2),
            'signals': signals
        }


    
    def z_score(self, stock_id, start_date, end_date):
        bs = self.data_service.get_data(stock_id, 'balance_sheet', start_date, end_date)
        fs = self.data_service.get_data(stock_id, 'financial_statement', start_date, end_date)

        # 取最新一季
        latest_date = bs['date'].max()
        bs_q = bs[bs['date'] == latest_date]
        fs_q = fs[fs['date'] == latest_date]

        def get_val(df, type_name):
            row = df[df['type'] == type_name]
            if row.empty:
                return None
            return row.iloc[0]['value']
        

        # 資產負債表
        current_assets = get_val(bs_q, 'CurrentAssets')
        current_liabilities = get_val(bs_q, 'CurrentLiabilities')
        total_assets = get_val(bs_q, 'TotalAssets')
        equity = get_val(bs_q, 'EquityAttributableToOwnersOfParent')
        retained_earnings = get_val(bs_q, 'RetainedEarnings')


        operating_income = get_val(fs_q, 'OperatingIncome')
        revenue = get_val(fs_q, 'Revenue')

        total_liabilities = total_assets - equity

        # A = 營運資金 / 總資產
        A = (current_assets - current_liabilities) / total_assets
        # B = 保留盈餘 / 總資產（沒有就用股東權益代替）
        B = (retained_earnings if retained_earnings else equity) / total_assets
        # C = 營業利益 / 總資產
        C = operating_income / total_assets
        # D = 跳過（需要市值，資料不易取得）
        D = 0
        # E = 營收 / 總資產
        E = revenue / total_assets

        z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E


        print(f'=== {stock_id} Altman Z-Score ===')
        print(f'報表日期: {latest_date}')
        print(f'A (營運資金/總資產): {A:.4f}')
        print(f'B (保留盈餘/總資產): {B:.4f}')
        print(f'C (營業利益/總資產): {C:.4f}')
        print(f'E (營收/總資產): {E:.4f}')
        print(f'Z-Score: {z:.2f}')

        if z > 2.99:
            print('判斷: 安全')
        elif z > 1.81:
            print('判斷: 灰色地帶（需注意）')
        else:
            print('判斷: 高風險')

        return round(z, 2)



        
        
