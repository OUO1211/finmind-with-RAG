"""
Backtester：回測引擎
功能：計算各種交易策略的歷史報酬率
"""

import pandas as pd


class Backtester:
    """
    回測引擎

    職責：
    1. 計算買進持有報酬率
    2. (未來) 支援自訂買賣策略
    """


    COMMISSION_RATE = 0.001425
    TAX_RATE = 0.003    

    def buy_and_hold(self, price_df: pd.DataFrame, include_cost: bool = True) -> dict:
        """
        計算買進持有報酬率

        策略：第一天收盤買入，最後一天收盤賣出

        Args:
            price_df: 股價 DataFrame，需包含 'date' 和 'close' 欄位

        Returns:
            dict: 包含買入價、賣出價、報酬率等資訊
        """
        df = price_df.sort_values('date').reset_index(drop=True)
        buy_price = df.iloc[0]['close']    # iloc[0] = 第一筆
        sell_price = df.iloc[-1]['close']  # iloc[-1] = 最後一筆

        if include_cost:
            actual_buy_cost =  buy_price * (1 + self.COMMISSION_RATE)
            actual_sell_income = sell_price * ( 1 - self.COMMISSION_RATE - self.TAX_RATE)
            returns = (actual_sell_income - actual_buy_cost) / actual_buy_cost * 100
        else:
            returns = (sell_price - buy_price) / buy_price * 100

        # Buy and Hold 全程持有，equity curve = 股價等比例縮放
        initial_capital = 1000000
        shares = initial_capital / (buy_price * (1 + self.COMMISSION_RATE))
        equity_df = df[['date']].copy()
        equity_df['close'] = shares * df['close']

        return {
            "buy_date": df.iloc[0]['date'],
            "sell_date": df.iloc[-1]['date'],
            "buy_price": buy_price,
            "sell_price": sell_price,
            "returns": round(returns, 2),
            "holding_days": len(df),
            "include_cost" : include_cost,
            "equity_curve": equity_df
        }   


    def pe_strategy(self, price_df: pd.DataFrame, per_df: pd.DataFrame,
                     buy_pe=15, sell_pe=25, include_cost: bool = True) -> dict:
        """
        PE 策略回測
        
        買入條件：PE < buy_pe
        賣出條件：PE > sell_pe
        
        Args:
            price_df: 股價 DataFrame
            per_df: PE 資料 DataFrame
            buy_pe: 買入門檻（PE 低於此值買入）
            sell_pe: 賣出門檻（PE 高於此值賣出）
            include_cost: 是否計入交易成本
        
        Returns:
            dict: 回測結果
        """

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        df = pd.merge(
            price_df[['date', 'close']],
            per_df[['date', 'PER']],
            on='date',
            how='inner'
        )
        
        df = df.sort_values('date').reset_index(drop=True)

        for i, row in df.iterrows():
            pe = row['PER']
            price = row['close']
            date = row['date']

            if not holding and pe < buy_pe:
                holding = True
                buy_price = price
                buy_date = date
                shares = cash / price
                cash = 0

            elif holding and pe > sell_pe:
                holding = False
                sell_price = price
                cash = shares * price
                shares = 0

                if include_cost:
                    cost = buy_price * (1 + self.COMMISSION_RATE)
                    income = sell_price * (1 - self.COMMISSION_RATE - self.TAX_RATE)
                    trade_return = (income - cost) / cost * 100
                else:
                    trade_return = (sell_price - buy_price) / buy_price * 100

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': round(trade_return, 2)
                })

            portfolio_value = cash + shares * price
            equity_data.append({'date': date, 'close': portfolio_value})

        equity_df = pd.DataFrame(equity_data)

        if len(trades) == 0:
            total_return = 0
        else:
            total_return = sum(t['return'] for t in trades)

        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_return': round(total_return, 2),
            'include_cost': include_cost,
            'equity_curve': equity_df
        }
        



    def ma_strategy(self, price_df: pd.DataFrame, short_window = 5, long_window = 20
                    ,include_cost: bool = True) -> dict:
        
        df = price_df.sort_values('date').reset_index(drop=True)
        df['short_ma'] = df['close'].rolling(short_window).mean()
        df['long_ma'] = df['close'].rolling(long_window).mean()
        df = df.dropna().reset_index(drop=True)

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            if not holding and row['short_ma'] > row['long_ma']:
                holding = True
                shares = cash / row['close']
                cash = 0
                buy_price = row['close']
                buy_date = row['date']

            elif holding and row['short_ma'] < row['long_ma']:
                holding = False
                cash = shares * row['close']
                shares = 0
                sell_price = row['close']

                if include_cost:
                    cost = buy_price * (1 + self.COMMISSION_RATE)
                    income = sell_price * (1 - self.COMMISSION_RATE - self.TAX_RATE)
                    trade_return = (income - cost) / cost * 100
                else:
                    trade_return = (sell_price - buy_price) / buy_price * 100

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': round(trade_return, 2)
                })

            portfolio_value = cash + shares * row['close']
            equity_data.append({'date': row['date'], 'close': portfolio_value})

        equity_df = pd.DataFrame(equity_data)

        if len(trades) == 0:
            total_return = 0
        else:
            total_return = sum(t['return'] for t in trades)

        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_return': round(total_return, 2),
            'include_cost': include_cost,
            'equity_curve': equity_df
        }
    

    def rsi_strategy(self, price_df: pd.DataFrame, period=14,
                 buy_rsi=30, sell_rsi=70, include_cost: bool = True) -> dict:
        df = price_df.sort_values('date').reset_index(drop=True)
        delta = df['close'].diff()           # 每天的漲跌幅
        gain = delta.clip(lower=0)           # 只取漲的（跌的變 0）
        loss = -delta.clip(upper=0)          # 只取跌的（漲的變 0）
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - 100 / (1 + rs)
        df = df.dropna().reset_index(drop=True)

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            if not holding and row['RSI'] < buy_rsi:
                holding = True
                buy_price = row['close']
                buy_date = row['date']
                shares = cash / row['close']
                cash = 0

            elif holding and row['RSI'] > sell_rsi:
                holding = False
                sell_price = row['close']
                cash = shares * row['close']
                shares = 0

                if include_cost:
                    cost = buy_price * (1 + self.COMMISSION_RATE)
                    income = sell_price * (1 - self.COMMISSION_RATE - self.TAX_RATE)
                    trade_return = (income - cost) / cost * 100
                else:
                    trade_return = (sell_price - buy_price) / buy_price * 100

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': round(trade_return, 2)
                })

            portfolio_value = cash + shares * row['close']
            equity_data.append({'date': row['date'], 'close': portfolio_value})

        equity_df = pd.DataFrame(equity_data)

        if len(trades) == 0:
            total_return = 0
        else:
            total_return = sum(t['return'] for t in trades)

        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_return': round(total_return, 2),
            'include_cost': include_cost,
            'equity_curve': equity_df
        }

    def kd_strategy(self, price_df: pd.DataFrame, period=9,
                    include_cost: bool = True) -> dict:
        df = price_df.sort_values('date').reset_index(drop=True)

        # 計算 RSV：收盤價在 9 天高低區間的位置（0~100）
        df['lowest'] = df['min'].rolling(period).min()
        df['highest'] = df['max'].rolling(period).max()
        df['RSV'] = (df['close'] - df['lowest']) / (df['highest'] - df['lowest']) * 100

        df = df.dropna().reset_index(drop=True)

        # 計算 K 和 D（遞迴，起始值 50）
        k_values = [50]
        d_values = [50]
        for i in range(len(df)):
            rsv = df.loc[i, 'RSV']
            k = k_values[-1] * 2/3 + rsv * 1/3
            d = d_values[-1] * 2/3 + k * 1/3
            k_values.append(k)
            d_values.append(d)
        df['K'] = k_values[1:]  
        df['D'] = d_values[1:]

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            if i == 0:
                equity_data.append({'date': row['date'], 'close': cash})
                continue  # 第一天沒有前一天可比較

            prev_k = df.loc[i-1, 'K']
            prev_d = df.loc[i-1, 'D']
            curr_k = row['K']
            curr_d = row['D']

            # 黃金交叉：K 從下往上穿越 D → 買
            if not holding and prev_k < prev_d and curr_k > curr_d:
                holding = True
                buy_price = row['close']
                buy_date = row['date']
                shares = cash / row['close']
                cash = 0

            # 死亡交叉：K 從上往下穿越 D → 賣
            elif holding and prev_k > prev_d and curr_k < curr_d:
                holding = False
                sell_price = row['close']
                cash = shares * row['close']
                shares = 0

                if include_cost:
                    cost = buy_price * (1 + self.COMMISSION_RATE)
                    income = sell_price * (1 - self.COMMISSION_RATE - self.TAX_RATE)
                    trade_return = (income - cost) / cost * 100
                else:
                    trade_return = (sell_price - buy_price) / buy_price * 100

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': round(trade_return, 2)
                })

            portfolio_value = cash + shares * row['close']
            equity_data.append({'date': row['date'], 'close': portfolio_value})

        equity_df = pd.DataFrame(equity_data)

        if len(trades) == 0:
            total_return = 0
        else:
            total_return = sum(t['return'] for t in trades)

        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_return': round(total_return, 2),
            'include_cost': include_cost,
            'equity_curve': equity_df
        }
