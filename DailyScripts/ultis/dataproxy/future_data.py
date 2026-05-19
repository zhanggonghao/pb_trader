import os
import traceback
import pandas as pd
import time
from datetime import datetime
from joblib import Parallel, delayed
from loguru import logger # type: ignore
import rqdatac # type: ignore
    
    
def update_future_data(data_output_path:str, start_date=None, end_date=None):
    if start_date is None:
        start_date = '2001-01-04'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = rqdatac.all_instruments(type='Future', market='cn', date=None)
    exchange_list = all_data['exchange'].unique().tolist()
    for exchange in exchange_list:
        if not os.path.exists(f"{data_output_path}/{exchange}"):
            os.makedirs(f"{data_output_path}/{exchange}")
        # 更新期货合约基础信息
        data = all_data[all_data['exchange'] == exchange]
        data.to_csv(f"{data_output_path}/{exchange}/future_basic.csv", index=False)
    # 更新期货合约1d数据
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Future', market='cn', date=trading_date)
        for exchange in exchange_list:
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if not os.path.exists(f"{data_output_path}/{exchange}/1d"):
                os.makedirs(f"{data_output_path}/{exchange}/1d")
            df = rqdatac.get_price(order_book_ids=ticker_list, frequency='1d', start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
            if df is None or df.empty is True:
                continue
            df = df.reset_index()
            df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d'))
            raw_columns = ['order_book_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_turnover', 'prev_close', 
                           'settlement', 'prev_settlement', 'open_interest', 'limit_up', 'limit_down', 'day_session_open']
            obj_columns = ["Ticker", "TradingDay", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", 
                           "Volume", "Turnover", "PreClosePrice", "SettlementPrice", "PreSettlementPrice",
                           "OpenInterest", "UpperLimitPrice", "LowerLimitPrice", "DayOpen"]
            data = pd.DataFrame(columns=obj_columns)
            data[obj_columns] = df[raw_columns]
            data['TradingDay'] = data['TradingDay'].astype(str)
            output_path = f"{data_output_path}/{exchange}/1d/{exchange}_Future_{trading_date.strftime('%Y%m%d')}.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"{output_path} save done")
            
    # 更新期货合约xmin数据
    if pd.to_datetime(start_date).date() < pd.to_datetime('2010-01-04').date():
        start_date = '2010-01-04'
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Future', market='cn', date=trading_date)
        if all_ticker_data is None or all_ticker_data.empty is True:
            continue
        for exchange in exchange_list:
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if len(ticker_list) == 0:
                continue
            frequency_list = ['1min', '5min', '15min', '30min', '60min']
            for frquency in frequency_list:
                if not os.path.exists(f"{data_output_path}/{exchange}/{frquency}"):
                    os.makedirs(f"{data_output_path}/{exchange}/{frquency}")
                df = rqdatac.get_price(order_book_ids=ticker_list, frequency=frquency[:-2], start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
                if df is None or df.empty is True:
                    continue
                df = df.reset_index()
                df['trading_date'] = df['trading_date'].apply(lambda x: x.strftime('%Y%m%d'))
                raw_columns = ['order_book_id', 'datetime', 'trading_date', 'open', 'high', 'low', 'close', 'volume', 'total_turnover', 
                               'open_interest']
                obj_columns = ["Ticker", "TimeStamp", "TradingDay", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", 
                               "Volume", "Turnover", "OpenInterest"]
                data = pd.DataFrame(columns=obj_columns)
                data[obj_columns] = df[raw_columns]
                data['TradingDay'] = data['TradingDay'].astype(str)
                output_path = f"{data_output_path}/{exchange}/{frquency}/{exchange}_Future_{trading_date.strftime('%Y%m%d')}.csv"
                data.to_csv(output_path, index=False)
                logger.info(f"{output_path} save done")

    
def update_option_data(data_output_path:str, start_date=None, end_date=None):
    if start_date is None:
        start_date = '2017-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    all_data = rqdatac.all_instruments(type='Option', market='cn', date=None)
    exchange_list = ["DCE", "SHFE", "CFFEX", "INE", "CZCE", "GFEX"]
    for exchange in exchange_list:
        if not os.path.exists(f"{data_output_path}/{exchange}"):
            os.makedirs(f"{data_output_path}/{exchange}")
        # 更新期权合约基础信息
        data = all_data[all_data['exchange'] == exchange]
        data.to_csv(f"{data_output_path}/{exchange}/option_basic.csv", index=False)
    # 更新期权合约1d数据
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Option', market='cn', date=trading_date)
        if all_ticker_data is None or all_ticker_data.empty is True:
            continue
        for exchange in exchange_list:
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if len(ticker_list) == 0:
                continue
            if not os.path.exists(f"{data_output_path}/{exchange}/1d"):
                os.makedirs(f"{data_output_path}/{exchange}/1d")
            df = rqdatac.get_price(order_book_ids=ticker_list, frequency='1d', start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
            if df is None or df.empty is True:
                continue
            df = df.reset_index()
            df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d'))
            raw_columns = ['order_book_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_turnover', 'prev_close', 
                           'settlement', 'prev_settlement', 'open_interest', 'limit_up', 'limit_down', 'day_session_open']
            obj_columns = ["Ticker", "TradingDay", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", 
                           "Volume", "Turnover", "PreClosePrice", "SettlementPrice", "PreSettlementPrice",
                           "OpenInterest", "UpperLimitPrice", "LowerLimitPrice", "DayOpen"]
            data = pd.DataFrame(columns=obj_columns)
            data[obj_columns] = df[raw_columns]
            data['TradingDay'] = data['TradingDay'].astype(str)
            output_path = f"{data_output_path}/{exchange}/1d/{exchange}_Option_{trading_date.strftime('%Y%m%d')}.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"{output_path} save done")
            
    # 更新期权合约xmin数据
    if pd.to_datetime(start_date).date() < pd.to_datetime('2017-01-01').date():
        start_date = '2017-01-01'
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Option', market='cn', date=trading_date)
        if all_ticker_data is None or all_ticker_data.empty is True:
            continue
        for exchange in exchange_list:
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if len(ticker_list) == 0:
                continue
            frequency_list = ['1min', '5min', '15min', '30min', '60min']
            for frquency in frequency_list:
                if not os.path.exists(f"{data_output_path}/{exchange}/{frquency}"):
                    os.makedirs(f"{data_output_path}/{exchange}/{frquency}")
                df = rqdatac.get_price(order_book_ids=ticker_list, frequency=frquency[:-2], start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
                if df is None or df.empty is True:
                    continue
                df = df.reset_index()
                df['trading_date'] = df['trading_date'].apply(lambda x: x.strftime('%Y%m%d'))
                raw_columns = ['order_book_id', 'datetime', 'trading_date', 'open', 'high', 'low', 'close', 'volume', 'total_turnover', 
                                'open_interest']
                obj_columns = ["Ticker", "TimeStamp", "TradingDay", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", 
                                "Volume", "Turnover", "OpenInterest"]
                data = pd.DataFrame(columns=obj_columns)
                data[obj_columns] = df[raw_columns]
                data['TradingDay'] = data['TradingDay'].astype(str)
                output_path = f"{data_output_path}/{exchange}/{frquency}/{exchange}_Option_{trading_date.strftime('%Y%m%d')}.csv"
                data.to_csv(output_path, index=False)
                logger.info(f"{output_path} save done")
    
    
def update_tick_data(data_output_path:str, start_date=None, end_date=None):
    raw_columns = ['order_book_id', 'timestamp', 'trading_date', 'actionday', 'updatetime', 'millsec', 'exchange', 'last', 
                   'volume', 'total_turnover', 'open', 'close', 'prev_close', 'settlement', 'prev_settlement', 
                   'open_interest', 'pre_open_interest', 'curr_delta', 'pre_delta', 'high', 'low', 
                   'limit_up', 'limit_down', 'avg_price', 'b1', 'b1_v', 'a1', 'a1_v', 
                   'b2', 'b2_v', 'a2', 'a2_v', 'b3', 'b3_v', 'a3', 'a3_v', 
                   'b4', 'b4_v', 'a4', 'a4_v', 'b5', 'b5_v', 'a5', 'a5_v']
    obj_columns = ["Ticker", "TimeStamp", "TradingDay", "ActionDay", "UpdateTime", "MillSec", "ExchangeID", "LastPrice", 
            "Volume", "Turnover", "OpenPrice", "ClosePrice", "PreClosePrice", "SettlementPrice", "PreSettlementPrice", 
            "OpenInterest", "PreOpenInterest",  "CurrDelta", "PreDelta", "HighestPrice", "LowestPrice", 
            "UpperLimitPrice", "LowerLimitPrice", "AveragePrice", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", 
            "BidPrice2", "BidVolume2", "AskPrice2", "AskVolume2", "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3", 
            "BidPrice4", "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5"]
    if start_date is None:
        start_date = '2010-01-04'
    if pd.to_datetime(start_date).date() < pd.to_datetime('2010-01-04').date():
        start_date = '2010-01-04'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    exchange_list = ["DCE", "SHFE", "CFFEX", "INE", "CZCE", "GFEX"]
    # 期货Level1数据
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Future', market='cn', date=trading_date)
        if all_ticker_data is None or all_ticker_data.empty is True:
            continue
        for exchange in exchange_list:
            if not os.path.exists(f"{data_output_path}/{exchange}/{exchange}_Level1"):
                os.makedirs(f"{data_output_path}/{exchange}/{exchange}_Level1")
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if len(ticker_list) == 0:
                continue
            df = rqdatac.get_price(order_book_ids=ticker_list, frequency='tick', start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
            if df is None or df.empty is True:
                continue
            df = df.reset_index()
            df['trading_date'] = df['trading_date'].apply(lambda x: x.date().strftime('%Y%m%d'))
            df['actionday'] = df['datetime'].apply(lambda x: x.date().strftime('%Y%m%d'))
            df['updatetime'] = df['datetime'].apply(lambda x: x.strftime('%H:%M:%S'))
            df['millsec'] = df['datetime'].apply(lambda x: x.microsecond/1000)
            df['timestamp'] = df['datetime'].apply(lambda x: time.mktime(x.timetuple())*1000 + x.microsecond/1000)
            df["exchange"] = exchange
            df["close"] = 0.0
            df["settlement"] = 0.0
            df["pre_open_interest"] = 0.0
            df["curr_delta"] = 0.0
            df["pre_delta"] = 0.0
            df['avg_price'] = 0.0
            data = pd.DataFrame(columns=obj_columns)
            data[obj_columns] = df[raw_columns]
            data['TradingDay'] = data['TradingDay'].astype(str)
            data['ActionDay'] = data['ActionDay'].astype(str)
            data['MillSec'] = data['MillSec'].astype(int)
            output_path = f"{data_output_path}/{exchange}/{exchange}_Level1/{exchange}_Future_Level1_{trading_date.strftime('%Y%m%d')}.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"{output_path} save done")
            
    # 期权Level1数据
    if start_date is None:
        start_date = '2017-01-01'
    if pd.to_datetime(start_date).date() < pd.to_datetime('2017-01-01').date():
        start_date = '2017-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    trading_date_list = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    for trading_date in trading_date_list:
        all_ticker_data = rqdatac.all_instruments(type='Option', market='cn', date=trading_date)
        if all_ticker_data is None or all_ticker_data.empty is True:
            continue
        for exchange in exchange_list:
            if not os.path.exists(f"{data_output_path}/{exchange}/{exchange}_Level1"):
                os.makedirs(f"{data_output_path}/{exchange}/{exchange}_Level1")
            ticker_list = all_ticker_data[all_ticker_data['exchange'] == exchange]['order_book_id'].unique().tolist()
            if len(ticker_list) == 0:
                continue
            df = rqdatac.get_price(order_book_ids=ticker_list, frequency='tick', start_date=trading_date, end_date=trading_date, adjust_type='post', expect_df=True)
            if df is None or df.empty is True:
                continue
            df = df.reset_index()
            df['trading_date'] = df['trading_date'].apply(lambda x: x.date().strftime('%Y%m%d'))
            df['actionday'] = df['datetime'].apply(lambda x: x.date().strftime('%Y%m%d'))
            df['updatetime'] = df['datetime'].apply(lambda x: x.strftime('%H:%M:%S'))
            df['millsec'] = df['datetime'].apply(lambda x: x.microsecond/1000)
            df['timestamp'] = df['datetime'].apply(lambda x: time.mktime(x.timetuple())*1000 + x.microsecond/1000)
            df["exchange"] = exchange
            df["close"] = 0.0
            df["settlement"] = 0.0
            df["pre_open_interest"] = 0.0
            df["curr_delta"] = 0.0
            df["pre_delta"] = 0.0
            df['avg_price'] = 0.0
            data = pd.DataFrame(columns=obj_columns)
            data[obj_columns] = df[raw_columns]
            data['TradingDay'] = data['TradingDay'].astype(str)
            data['ActionDay'] = data['ActionDay'].astype(str)
            data['MillSec'] = data['MillSec'].astype(int)
            output_path = f"{data_output_path}/{exchange}/{exchange}_Level1/{exchange}_Option_Level1_{trading_date.strftime('%Y%m%d')}.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"{output_path} save done")


def update_benchmark_data(data_output_path:str, start_date=None, end_date=None):
    if start_date is None:
        start_date = '2001-01-04'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    order_book_id = '000012.XSHG'
    # 更新000012.SH 1d数据
    if not os.path.exists(f"{data_output_path}/CFFEX/1d"):
        os.makedirs(f"{data_output_path}/CFFEX/1d")
    df = rqdatac.get_price(order_book_ids=order_book_id, frequency='1d', start_date='2001-01-04', end_date=end_date, adjust_type='post', expect_df=True)
    if df is not None or df.empty is False:
        df = df.reset_index()
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d'))
        raw_columns = ['order_book_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'total_turnover', 'prev_close']
        obj_columns = ["Ticker", "TradingDay", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", 
                        "Volume", "Turnover", "PreClosePrice", "SettlementPrice", "PreSettlementPrice",
                        "OpenInterest", "UpperLimitPrice", "LowerLimitPrice", "DayOpen"]
        data = pd.DataFrame(columns=obj_columns)
        data["Ticker"] = df['order_book_id']
        data["TradingDay"] = df['date']
        data["OpenPrice"] = df['open']
        data["HighestPrice"] = df['high']
        data["LowestPrice"] = df['low']
        data["ClosePrice"] = df['close']
        data["Volume"] = df['volume']
        data["Turnover"] = df['total_turnover']
        data["PreClosePrice"] = df['prev_close']
        data["OpenInterest"] = 0
        data['TradingDay'] = data['TradingDay'].astype(str)
        output_path = f"{data_output_path}/CFFEX/1d/{order_book_id}.csv"
        data.to_csv(output_path, index=False)
        logger.info(f"{output_path} save done")
        
    # 更新000012.SH xmin数据
    if pd.to_datetime(start_date).date() < pd.to_datetime('2010-01-04').date():
        start_date = '2010-01-04'
    frequency_list = ['1min', '5min', '15min', '30min', '60min']
    for frquency in frequency_list:
        if not os.path.exists(f"{data_output_path}/CFFEX/{frquency}"):
            os.makedirs(f"{data_output_path}/CFFEX/{frquency}")
        output_path = f"{data_output_path}/CFFEX/{frquency}/{order_book_id}.csv"
        if os.path.exists(output_path):
            df_local = pd.read_csv(output_path)
            df_local.set_index(['Ticker', 'TimeStamp'], inplace=True)
            start_date = rqdatac.get_next_trading_date(df_local.index.get_level_values('TimeStamp').max())
        else:
            df_local = None
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            logger.info(f"{order_book_id}当前已是最新数据，跳过 {start_date} to {end_date}")
            return 

        logger.info(f"正在下载{order_book_id}数据，{start_date} to {end_date}")
        df = rqdatac.get_price(order_book_ids=order_book_id, frequency=frquency[:-2], start_date=start_date, end_date=end_date, adjust_type='post', expect_df=True)
        if df is None or df.empty is True:
            continue
        df = df.reset_index()
        raw_columns = ['order_book_id', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'total_turnover']
        obj_columns = ["Ticker", "TimeStamp", "OpenPrice", "HighestPrice", "LowestPrice", "ClosePrice", "Volume", "Turnover"]
        data = pd.DataFrame()
        data[obj_columns] = df[raw_columns]
        data['TradingDay'] = data['TimeStamp'].apply(lambda x: pd.to_datetime(x).date())
        data['OpenInterest'] = 0
        data['TradingDay'] = data['TradingDay'].astype(str)
        data.set_index(['Ticker', 'TimeStamp'], inplace=True)
        if df_local is None:
            df_all = data
        else:
            df_all = pd.concat([df_local, data]).sort_index()
        df_all = df_all.reset_index()
        df_all.to_csv(output_path, index=False)
        logger.info(f"{output_path} save done")