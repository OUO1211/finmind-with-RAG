from src.data_service import DataService
from src.out_of_sample import OutOfSampleValidator

data_service = DataService()
price_df = data_service.get_data(
    "2330",
    "stock_price",
    "2019-01-01",
    "2026-12-31",
)

report = OutOfSampleValidator().evaluate(
    "buy_and_hold",
    price_df,
)

print(report["comparison"])

