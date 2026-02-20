from src.data_service import DataService
import matplotlib.pyplot as plt



class CashFlowAnalyzer:
    def __init__(self):
        self.dataservice = DataService()
        
    def get_summary(self, stock_id, start_date, end_date):
        cf_df = self.dataservice.get_data(stock_id, 'cash_flow', start_date, end_date)
        target_types = [
            'CashFlowsFromOperatingActivities',
            'CashProvidedByInvestingActivities',
            'CashFlowsProvidedFromFinancingActivities'
        ]
        filtered_df = cf_df[cf_df['type'].isin(target_types)]
        summary = filtered_df.pivot_table(index='date', columns='type', values='value')
        summary.columns = ['營業活動（億元）', '籌資活動（億元）', '投資活動（億元）']
        summary = summary / 1e8
        summary = summary.round(2)
        return summary

    def free_cash_flow(self, stock_id, start_date, end_date):
        # 1. 取得現金流量資料
        cf_df = self.dataservice.get_data(stock_id, 'cash_flow', start_date, end_date)
        
        # 2. 篩選營業活動 + 資本支出
        target_types = [
            'CashFlowsFromOperatingActivities',
            'PropertyAndPlantAndEquipment'
        ]
        filtered_df = cf_df[cf_df['type'].isin(target_types)]
        
        # 3. 轉成寬格式
        summary = filtered_df.pivot_table(index='date', columns='type', values='value')
        
        # 4. 改欄位名稱
        summary.columns = ['營業活動', '資本支出']
        
        # 5. 計算自由現金流（營業活動 + 資本支出，因為資本支出已經是負數）
        summary['自由現金流'] = summary['營業活動'] + summary['資本支出']
        
        # 6. 轉億元
        summary = summary / 1e8
        summary = summary.round(2)
        
        return summary
    
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    def plot_summary(self, stock_id, start_date, end_date):
        summary = self.get_summary(stock_id, start_date, end_date)
        summary.plot(kind='bar', figsize=(12, 6))
        plt.title('現金流量摘要')
        plt.ylabel('億元')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.show()

    
    def plot_fcf(self, stock_id, start_date, end_date):
        fcf = self.free_cash_flow(stock_id, start_date, end_date)
        colors = ['green' if v >= 0 else 'red' for v in fcf['自由現金流']]
        fcf['自由現金流'].plot(kind='bar', color=colors, figsize=(12, 6))
        plt.title('自由現金流趨勢')
        plt.ylabel('億元')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.axhline(y=0, color='black', linewidth=0.5)
        plt.show()

