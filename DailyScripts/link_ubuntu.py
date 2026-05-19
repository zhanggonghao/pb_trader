
from ultis.stock_data_client import StockDataClient
client = StockDataClient(data_path=r'Z:\Market')

data = client.get_stock_index_comments_weights('000300.XSHG', start='2026-05-06', end='2026-05-06')

print(data)
