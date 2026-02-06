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

        return {
            "buy_date": df.iloc[0]['date'],
            "sell_date": df.iloc[-1]['date'],
            "buy_price": buy_price,
            "sell_price": sell_price,
            "returns": round(returns, 2),  # 報酬率，保留兩位小數
            "holding_days": len(df),       # 持有天數
            "include_cost" : include_cost
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
                # 沒持有 + PE 低於門檻 -> 買入
                holding = True
                buy_price = price
                buy_date = date

            
            elif holding and pe > sell_pe:
                # 持有中 + PE 高於門檻 → 賣出
                holding = False
                sell_price = price

                # 計算這筆交易的報酬
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

        if len(trades) == 0:
            total_return = 0
        else:
            total_return = sum(t['return'] for t in trades)

        return {
            'trades': trades,
            'total_trades': len(trades),
            'total_return': round(total_return, 2),
            'include_cost': include_cost
        }
        


