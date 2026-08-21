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
    # 預設維持舊回測結果；啟用後可用固定比例模擬每一邊的委託衝擊。
    # 例如 slippage_rate=0.001 時，市價買入 100 元會以 100.10 元成交。
    DEFAULT_SLIPPAGE_RATE = 0.0

    def __init__(self, slippage_rate: float = DEFAULT_SLIPPAGE_RATE):
        if slippage_rate < 0:
            raise ValueError("slippage_rate must be non-negative")
        self.slippage_rate = slippage_rate

    def _fill_price(self, market_price: float, side: str, include_cost: bool) -> float:
        """Return the adverse fill price for a buy or sell order."""
        if not include_cost:
            return market_price
        if side == 'buy':
            return market_price * (1 + self.slippage_rate)
        if side == 'sell':
            return market_price * (1 - self.slippage_rate)
        raise ValueError("side must be 'buy' or 'sell'")

    def _buy_cash_per_share(self, market_price: float, include_cost: bool) -> float:
        fill_price = self._fill_price(market_price, 'buy', include_cost)
        return fill_price * (1 + self.COMMISSION_RATE) if include_cost else fill_price

    def _sell_cash_per_share(self, market_price: float, include_cost: bool) -> float:
        fill_price = self._fill_price(market_price, 'sell', include_cost)
        return fill_price * (1 - self.COMMISSION_RATE - self.TAX_RATE) if include_cost else fill_price

    def _trade_return(self, buy_price: float, sell_price: float, include_cost: bool) -> float:
        cost = self._buy_cash_per_share(buy_price, include_cost)
        income = self._sell_cash_per_share(sell_price, include_cost)
        return (income - cost) / cost * 100

    def _position_shares(self, cash: float, market_price: float, include_cost: bool) -> float:
        """Keep legacy equity curves unchanged until slippage is explicitly enabled."""
        if not include_cost or self.slippage_rate == 0:
            return cash / market_price
        return cash / self._buy_cash_per_share(market_price, include_cost)

    def _liquidation_cash(self, shares: float, market_price: float, include_cost: bool) -> float:
        if not include_cost or self.slippage_rate == 0:
            return shares * market_price
        return shares * self._sell_cash_per_share(market_price, include_cost)

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

        returns = self._trade_return(buy_price, sell_price, include_cost)

        # Buy and Hold 全程持有，equity curve = 股價等比例縮放
        initial_capital = 1000000
        shares = initial_capital / self._buy_cash_per_share(buy_price, include_cost)
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
            "slippage_rate": self.slippage_rate if include_cost else 0.0,
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
            price_df[['date', 'open', 'close']],
            per_df[['date', 'PER']],
            on='date',
            how='inner'
        )

        df = df.sort_values('date').reset_index(drop=True)

        # 訊號與執行分離：見 ma_strategy 的說明，這裡同樣套用
        buy_signal = df['PER'] < buy_pe
        sell_signal = df['PER'] > sell_pe
        df['execute_buy'] = buy_signal.shift(1, fill_value=False)
        df['execute_sell'] = sell_signal.shift(1, fill_value=False)

        for i, row in df.iterrows():
            price = row['close']
            date = row['date']

            if not holding and row['execute_buy']:
                holding = True
                buy_price = row['open']
                buy_date = date
                shares = self._position_shares(cash, buy_price, include_cost)
                cash = 0

            elif holding and row['execute_sell']:
                holding = False
                sell_price = row['open']
                cash = self._liquidation_cash(shares, sell_price, include_cost)
                shares = 0

                trade_return = self._trade_return(buy_price, sell_price, include_cost)

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'buy_fill_price': self._fill_price(buy_price, 'buy', include_cost),
                    'sell_fill_price': self._fill_price(sell_price, 'sell', include_cost),
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
            'slippage_rate': self.slippage_rate if include_cost else 0.0,
            'equity_curve': equity_df
        }
        



    def ma_strategy(self, price_df: pd.DataFrame, short_window = 5, long_window = 20
                    ,include_cost: bool = True) -> dict:

        df = price_df.sort_values('date').reset_index(drop=True)
        df['short_ma'] = df['close'].rolling(short_window).mean()
        df['long_ma'] = df['close'].rolling(long_window).mean()
        df = df.dropna().reset_index(drop=True)

        # 訊號與執行分離：訊號用「當天收盤」算出來（向量化，不用等迴圈跑到那一列），
        # 但 .shift(1) 把整欄往下移一格，讓「今天」這一列看到的是「昨天」算出的
        # 訊號——對應真實世界「收盤後才知道訊號、最早只能隔天開盤成交」的限制。
        # 第一列位移後沒有「昨天」可參考，用 fill_value=False 代表當天不執行
        # （直接讓 shift 補值，型別維持 bool，不會像 shift 後再 fillna 那樣
        # 中間產生 NaN 導致整欄被降級成 object dtype）。
        buy_signal = df['short_ma'] > df['long_ma']
        sell_signal = df['short_ma'] < df['long_ma']
        df['execute_buy'] = buy_signal.shift(1, fill_value=False)
        df['execute_sell'] = sell_signal.shift(1, fill_value=False)

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            if not holding and row['execute_buy']:
                holding = True
                shares = self._position_shares(cash, row['open'], include_cost)
                cash = 0
                buy_price = row['open']
                buy_date = row['date']

            elif holding and row['execute_sell']:
                holding = False
                cash = self._liquidation_cash(shares, row['open'], include_cost)
                shares = 0
                sell_price = row['open']

                trade_return = self._trade_return(buy_price, sell_price, include_cost)

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'buy_fill_price': self._fill_price(buy_price, 'buy', include_cost),
                    'sell_fill_price': self._fill_price(sell_price, 'sell', include_cost),
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
            'slippage_rate': self.slippage_rate if include_cost else 0.0,
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

        # 訊號與執行分離：見 ma_strategy 的說明，這裡同樣套用
        buy_signal = df['RSI'] < buy_rsi
        sell_signal = df['RSI'] > sell_rsi
        df['execute_buy'] = buy_signal.shift(1, fill_value=False)
        df['execute_sell'] = sell_signal.shift(1, fill_value=False)

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            if not holding and row['execute_buy']:
                holding = True
                buy_price = row['open']
                buy_date = row['date']
                shares = self._position_shares(cash, buy_price, include_cost)
                cash = 0

            elif holding and row['execute_sell']:
                holding = False
                sell_price = row['open']
                cash = self._liquidation_cash(shares, sell_price, include_cost)
                shares = 0

                trade_return = self._trade_return(buy_price, sell_price, include_cost)

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'buy_fill_price': self._fill_price(buy_price, 'buy', include_cost),
                    'sell_fill_price': self._fill_price(sell_price, 'sell', include_cost),
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
            'slippage_rate': self.slippage_rate if include_cost else 0.0,
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

        # 黃金/死亡交叉本身就要比較「昨天」跟「今天」的 K/D，用 .shift(1) 先把
        # 「昨天的 K/D」攤平成獨立欄位，向量化算出交叉訊號（不用在迴圈裡查
        # df.loc[i-1, ...]）。算出交叉訊號後，再用 .shift(1) 做「訊號與執行分離」
        # （見 ma_strategy 說明）——這裡等於位移了兩次：第一次是交叉判斷本身需要
        # 的「昨天 vs 今天」比較，第二次是「今天判斷出訊號、隔天才能執行」。
        prev_k = df['K'].shift(1)
        prev_d = df['D'].shift(1)
        buy_signal = (prev_k < prev_d) & (df['K'] > df['D'])
        sell_signal = (prev_k > prev_d) & (df['K'] < df['D'])
        df['execute_buy'] = buy_signal.shift(1, fill_value=False)
        df['execute_sell'] = sell_signal.shift(1, fill_value=False)

        holding = False
        buy_price = 0
        buy_date = None
        trades = []
        initial_capital = 1000000
        cash = initial_capital
        shares = 0
        equity_data = []

        for i, row in df.iterrows():
            # 黃金交叉：K 從下往上穿越 D → 買
            if not holding and row['execute_buy']:
                holding = True
                buy_price = row['open']
                buy_date = row['date']
                shares = self._position_shares(cash, buy_price, include_cost)
                cash = 0

            # 死亡交叉：K 從上往下穿越 D → 賣
            elif holding and row['execute_sell']:
                holding = False
                sell_price = row['open']
                cash = self._liquidation_cash(shares, sell_price, include_cost)
                shares = 0

                trade_return = self._trade_return(buy_price, sell_price, include_cost)

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': row['date'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'buy_fill_price': self._fill_price(buy_price, 'buy', include_cost),
                    'sell_fill_price': self._fill_price(sell_price, 'sell', include_cost),
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
            'slippage_rate': self.slippage_rate if include_cost else 0.0,
            'equity_curve': equity_df
        }
