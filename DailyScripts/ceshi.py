
from ultis.stock_data_client import StockDataClient
client = StockDataClient(data_path='/home/samba/Market')

pre_date = '2026-04-21'
sample_range_df = client.get_stock_index_comments('zz1800', start=pre_date, end=pre_date).T
sample_range_df = sample_range_df[sample_range_df[pre_date] == 1]
print(sample_range_df)
print(sample_range_df.index.tolist())