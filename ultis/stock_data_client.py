import pandas as pd
import os
import copy
import numpy as np
import datetime
from joblib import Parallel, delayed

class StockDataClient(object):
    STOCK_BASIC_INFO_FILEPATH = 'stocks/basics/basic_info.feather'
    STOCK_INDUSTRY_FILEPATH = 'stocks/basics/all_instruments_level1_industry.feather'
    ST_STOCK_FILEPATH = "stocks/basics/st_stock.feather"
    SUSPENDED_STOCK_FILEPATH = "stocks/basics/suspended_stock.feather"
    INDEX_COMPONENTS_FILEPATH = "stocks/basics/index_components_{order_book_id}.feather"
    INDEX_COMPONENTS_WEIGHTS_FILEPATH = "stocks/basics/index_components_weights_{order_book_id}.feather"
    INDEX_COMPONENTS_WEIGHTS_INDUSTRY_FILEPATH = "stocks/basics/index_components_weights_industry_{order_book_id}.feather"
    STOCK_POST_1D_FILEPATH = "stocks/stocks_post_1d/stocks_post_1d.feather"
    STOCK_POST_VWAP_1D_FILEPATH = "stocks/stocks_post_1d/stocks_vwap_post_1d.feather"
    STOCK_POST_TWAP_1D_FILEPATH = "stocks/stocks_post_1d/stocks_twap_post_1d.feather"
    STOCK_POST_1MIN_DATAPATH = "stocks/stocks_post_1min"
    STOCK_POST_XMIN_DATAPATH = "stocks/stocks_post_{frequency}"
    
    FUTURE_POST_1MIN_DATAPATH = "futures/futures_post_1min"
    FUTURE_POST_1D_DATAPATH = "futures/futures_post_1d"
    FUTURE_STATE_MATRIX_FILEPATH = 'futures/futures_post_1d/index_state_matrix.feather'
    FUTURE_MATURITY_DATE_FILEPATH = 'futures/futures_post_1d/index_maturity_date.feather'
    FUTURE_CONSISTENT_PRICES_FILEPATH = 'futures/futures_post_1d/index_futures_daily_consistent_prices.feather'
    FUTURE_XMIN_CONSISTENT_PRICES_FILEPATH = 'futures/futures_post_1min/index_futures_xmin_consistent_prices.feather'
    
    FACTOR_POST_1D_DATAPATH = "factors/factors_post_1d"
    FACTOR_POST_XMIN_DATAPATH = "factors/factors_post_{frequency}" 
    
    RQFACTOR_POST_1D_DATAPATH = "rqfactors/factors_post_1d"
    
    def __init__(self, data_path:str) -> None:
        self.basic_info_filepath = StockDataClient.STOCK_BASIC_INFO_FILEPATH
        self.all_instruments_industry_path = StockDataClient.STOCK_INDUSTRY_FILEPATH 
        self.st_stock_filepath = StockDataClient.ST_STOCK_FILEPATH
        self.suspended_stock_filepath = StockDataClient.SUSPENDED_STOCK_FILEPATH
        self.index_comments_filepath = StockDataClient.INDEX_COMPONENTS_FILEPATH
        self.index_comments_weights_filepath = StockDataClient.INDEX_COMPONENTS_WEIGHTS_FILEPATH
        self.index_comments_weights_industry_filepath = StockDataClient.INDEX_COMPONENTS_WEIGHTS_INDUSTRY_FILEPATH
        self.stock_post_1d_filepath = StockDataClient.STOCK_POST_1D_FILEPATH
        self.stock_post_vwap_1d_filepath = StockDataClient.STOCK_POST_VWAP_1D_FILEPATH
        self.stock_post_twap_1d_filepath = StockDataClient.STOCK_POST_TWAP_1D_FILEPATH
        self.stock_post_1min_datapath = StockDataClient.STOCK_POST_1MIN_DATAPATH
        self.stock_post_xmin_datapath = StockDataClient.STOCK_POST_XMIN_DATAPATH
        
        self.future_post_1d_datapath = StockDataClient.FUTURE_POST_1D_DATAPATH
        self.future_post_1min_datapath = StockDataClient.FUTURE_POST_1MIN_DATAPATH
        self.future_state_matrix_filepath = StockDataClient.FUTURE_STATE_MATRIX_FILEPATH
        self.future_maturity_date_filepath = StockDataClient.FUTURE_MATURITY_DATE_FILEPATH
        self.future_consistent_prices_filepath = StockDataClient.FUTURE_CONSISTENT_PRICES_FILEPATH
        self.future_xmin_consistent_prices_filepath = StockDataClient.FUTURE_XMIN_CONSISTENT_PRICES_FILEPATH
        self.factor_post_xmin_datapath = StockDataClient.FACTOR_POST_XMIN_DATAPATH
        self.factor_post_1d_datapath = StockDataClient.FACTOR_POST_1D_DATAPATH
        
        self.rqfactor_post_1d_datapath = StockDataClient.RQFACTOR_POST_1D_DATAPATH
        
        self.data_path = data_path
    
    def get_stock_basic_info(self)->pd.DataFrame:
        """
        获取股票基础信息数据
        Raises:
            FileNotFoundError:文件数据不存在则抛出异常
        Returns:
            pd.DataFrame: 返回数据
        """
        filepath = os.path.join(self.data_path, self.basic_info_filepath)
        if os.path.exists(filepath):
            df = pd.read_feather(filepath)
            df.set_index(['order_book_id'], inplace=True)
        else:
            df = None
            raise FileNotFoundError(f"{filepath} not found")
        return df

    def get_all_instruments_industry_data(self)->pd.DataFrame:
        """
        获取股票行业数据
        Raises:
            FileNotFoundError:文件数据不存在则抛出异常
        Returns:
            pd.DataFrame: 返回数据
        """
        filepath = os.path.join(self.data_path, self.all_instruments_industry_path)
        if os.path.exists(filepath):
            df = pd.read_feather(filepath)
            df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
            df.set_index(['date', 'order_book_id'], inplace=True)
        else:
            df = None
            raise FileNotFoundError(f"{filepath} not found")
        return df
    
    def get_stock_st_info(self, order_book_ids=None, start=None, end=None)->pd.DataFrame:
        """
        获取股票ST信息
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.st_stock_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date'], inplace=True)
        if start:
            df = df[start:]
        if end:
            df = df[:end]
        if order_book_ids:
            df = df.reindex(columns=order_book_ids)
        return df
    
    def get_stock_suspended_info(self, order_book_ids=None, start=None, end=None)->pd.DataFrame:
        """
        获取股票停牌信息
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.suspended_stock_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date'], inplace=True)
        if start:
            df = df[start:]
        if end:
            df = df[:end]
        if order_book_ids:
            df = df.reindex(columns=order_book_ids)
        df.replace(0.0, np.nan)
        return df
    
    def get_stock_index_comments(self, order_book_id:str, start=None, end=None)->pd.DataFrame:
        """
        获取指数成分信息
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.index_comments_filepath.format(order_book_id=order_book_id))
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date'], inplace=True)
        if start:
            df = df[start:]
        if end:
            df = df[:end]
        df = df.replace(0.0, np.nan)
        return df
    
    def get_stock_index_comments_weights(self, order_book_id:str, start=None, end=None)->pd.DataFrame:
        """
        获取指数成分权重信息
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.index_comments_weights_filepath.format(order_book_id=order_book_id))
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date'], inplace=True)
        if start:
            df = df[start:]
        if end:
            df = df[:end]
        return df

    def get_stock_index_comments_weights_industry(self, order_book_id:str, start=None, end=None)->pd.DataFrame:
        """
        获取指数成分权重行业信息
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.index_comments_weights_industry_filepath.format(order_book_id=order_book_id))
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['date', 'order_book_id'], inplace=True)
        df = df.sort_index()
        if start:
            df = df[start:]
        if end:
            df = df[:end]
        return df
    
    def get_stock_post_1d_data(self, order_book_ids=None, start=None, end=None)->pd.DataFrame:
        """
        获取日度行情数据
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.stock_post_1d_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['order_book_id', 'date'], inplace=True)
        if order_book_ids:
            df = df[df.index.get_level_values('order_book_id').isin(order_book_ids)]
        if start and end:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start) & (df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        elif start:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start)].set_index(['order_book_id', 'date']).sort_index()
        elif end:
            df.reset_index(inplace=True)
            df = df[(df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        return df
    
    def get_stock_post_vwap_1d_data(self, order_book_ids=None, start=None, end=None)->pd.DataFrame:
        """
        获取日度VWAP行情数据
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.stock_post_vwap_1d_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['order_book_id', 'date'], inplace=True)
        if order_book_ids:
            df = df[df.index.get_level_values('order_book_id').isin(order_book_ids)]
        if start and end:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start) & (df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        elif start:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start)].set_index(['order_book_id', 'date']).sort_index()
        elif end:
            df.reset_index(inplace=True)
            df = df[(df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        return df
    
    def get_stock_post_twap_1d_data(self, order_book_ids=None, start=None, end=None)->pd.DataFrame:
        """
        获取日度TWAP行情数据
        Args:
            order_book_ids: 股票过滤列表,默认为None,表示提取全部标的。
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.stock_post_twap_1d_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['order_book_id', 'date'], inplace=True)
        if order_book_ids:
            df = df[df.index.get_level_values('order_book_id').isin(order_book_ids)]
        if start and end:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start) & (df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        elif start:
            df.reset_index(inplace=True)
            df = df[(df['date'] >= start)].set_index(['order_book_id', 'date']).sort_index()
        elif end:
            df.reset_index(inplace=True)
            df = df[(df['date'] <= end)].set_index(['order_book_id', 'date']).sort_index()
        return df
    
    def get_stock_post_1min_data(self, order_book_ids:list, start=None, end=None, parallel=True)->pd.DataFrame:
        """
        获取1min级别行情数据
        Args:
            order_book_ids: 股票过滤列表
            start:起始时间,默认为None,不进行过滤
            end:结束时间,默认为None,不进行过滤
            parallel:是否支持并行,默认为True

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if not parallel:
            df_list = list()
            for order_book_id in order_book_ids:
                filepath = os.path.join(self.data_path, self.stock_post_1min_datapath, f"{order_book_id}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index(['datetime'], inplace=True)
                data.reset_index(inplace=True)
                data['order_book_id'] = order_book_id
                if start and end:
                    data = data[(data['datetime'] >= start) & (data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                elif start:
                    data = data[(data['datetime'] >= start)].set_index(['order_book_id', 'datetime']).sort_index()
                elif end:
                    data = data[(data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                else:
                    data = data.set_index(['order_book_id', 'datetime']).sort_index()
                df_list.append(data)
            df = pd.concat(df_list)
        else:
            def _get_stock_post_1min_data(order_book_id, start, end):
                filepath = os.path.join(self.data_path, self.stock_post_1min_datapath, f"{order_book_id}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index(['datetime'], inplace=True)
                data.reset_index(inplace=True)
                data['order_book_id'] = order_book_id
                if start and end:
                    data = data[(data['datetime'] >= start) & (data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                elif start:
                    data = data[(data['datetime'] >= start)].set_index(['order_book_id', 'datetime']).sort_index()
                elif end:
                    data = data[(data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                else:
                    data = data.set_index(['order_book_id', 'datetime']).sort_index()
                return data
            
            jobs = len(order_book_ids)
            if jobs > 128:
                jobs = 128
            df_list = Parallel(n_jobs=jobs)(delayed(_get_stock_post_1min_data)(order_book_id, start, end) for order_book_id in order_book_ids)
            df = pd.concat(df_list)
        return df
    
    def get_stock_post_vwap_twap_xmin_data(self, frequency:str, order_book_ids:list, start=None, end=None, parallel=True)->pd.DataFrame:
        """
        获取分钟级别VWAP/TWAP行情数据
        Args:
            frequency: 频率， 5min,15min,30min,60min
            order_book_ids: 股票过滤列表
            start:起始时间,默认为None,不进行过滤
            end:结束时间,默认为None,不进行过滤
            parallel:是否支持并行,默认为True

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        stock_post_xmin_datapath = self.stock_post_xmin_datapath.format(frequency=frequency)
        if not parallel:
            df_list = list()
            for order_book_id in order_book_ids:
                filepath = os.path.join(self.data_path, stock_post_xmin_datapath, f"{order_book_id}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index(['datetime'], inplace=True)
                data.reset_index(inplace=True)
                data['order_book_id'] = order_book_id
                if start and end:
                    data = data[(data['datetime'] >= start) & (data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                elif start:
                    data = data[(data['datetime'] >= start)].set_index(['order_book_id', 'datetime']).sort_index()
                elif end:
                    data = data[(data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                else:
                    data = data.set_index(['order_book_id', 'datetime']).sort_index()
                df_list.append(data)
            df = pd.concat(df_list)
        else:
            def _get_stock_post_vwap_twap_xmin_data(order_book_id, start, end):
                filepath = os.path.join(self.data_path, stock_post_xmin_datapath, f"{order_book_id}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index(['datetime'], inplace=True)
                data.reset_index(inplace=True)
                data['order_book_id'] = order_book_id
                if start and end:
                    data = data[(data['datetime'] >= start) & (data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                elif start:
                    data = data[(data['datetime'] >= start)].set_index(['order_book_id', 'datetime']).sort_index()
                elif end:
                    data = data[(data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
                else:
                    data = data.set_index(['order_book_id', 'datetime']).sort_index()
                return data
            
            jobs = len(order_book_ids)
            if jobs > 128:
                jobs = 128
            df_list = Parallel(n_jobs=jobs)(delayed(_get_stock_post_vwap_twap_xmin_data)(order_book_id, start, end) for order_book_id in order_book_ids)
            df = pd.concat(df_list)
        return df
    
    def get_future_post_1min_data(self, order_book_ids:list, start=None, end=None)->pd.DataFrame:
        """
        获取1分钟级别期货行情数据
        Args:
            order_book_ids: 股指合约过滤列表
            start:起始时间,默认为None,不进行过滤
            end:结束时间,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df_list = list()
        for order_book_id in order_book_ids:
            filepath = os.path.join(self.data_path, self.future_post_1min_datapath, f"{order_book_id}.feather")
            data = pd.read_feather(filepath)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index(['datetime'], inplace=True)
            data.reset_index(inplace=True)
            data['order_book_id'] = order_book_id
            if start and end:
                data = data[(data['datetime'] >= start) & (data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
            elif start:
                data = data[(data['datetime'] >= start)].set_index(['order_book_id', 'datetime']).sort_index()
            elif end:
                data = data[(data['datetime'] <= end)].set_index(['order_book_id', 'datetime']).sort_index()
            else:
                data = data.set_index(['order_book_id', 'datetime']).sort_index()
            df_list.append(data)
        df = pd.concat(df_list)
        return df
    
    def get_future_state_matrix_data(self, products:list, start=None, end=None)->pd.DataFrame:
        """
        获取股指期货合约状态信息
        Args:
            products: 股指合约品种过滤列表
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.future_state_matrix_filepath)
        data = pd.read_feather(filepath)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index(['date'], inplace=True)
        data.index.name = 'date'
        data.columns.name = None
        df_list = list()
        for product in products:
            df1 = data[data['product'] == product]
            if start and end:
                df1 = df1[(df1.index >= start) & (df1.index <= end)].sort_index()
            elif start:
                df1 = df1[(df1.index >= start)].sort_index()
            elif end:
                df1 = df1[(df1.index <= end)].sort_index()
            else:
                df1 = df1.sort_index()
            df_list.append(df1)
        df = pd.concat(df_list)
        return df
    
    def get_future_maturity_date_data(self, products:list, start=None, end=None)->pd.DataFrame:
        """
        获取股指期货合约到期日信息
        Args:
            products: 股指合约品种过滤列表
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.future_maturity_date_filepath)
        data = pd.read_feather(filepath)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index(['date'], inplace=True)
        data.index.name = 'date'
        data.columns.name = None
        df_list = list()
        for product in products:
            df1 = data[data['product'] == product]
            if start and end:
                df1 = df1[(df1.index >= start) & (df1.index <= end)].sort_index()
            elif start:
                df1 = df1[(df1.index >= start)].sort_index()
            elif end:
                df1 = df1[(df1.index <= end)].sort_index()
            else:
                df1 = df1.sort_index()
            df1 = df1.reindex(columns=[f"{product}_CM", f"{product}_NM", f"{product}_CQ", f"{product}_NQ"])
            df_list.append(df1)
        df = pd.concat(df_list)
        return df
    
    def get_index_consistent_prices(self, order_book_ids:list, start=None, end=None):
        """
        获取股指期货日度连续行情
        Args:
            order_book_ids: 股指合约过滤列表
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.future_consistent_prices_filepath)
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(['order_book_id', 'date'], inplace=True)
        # print(df)
        if order_book_ids:
            df = df[df.index.get_level_values('order_book_id').isin(order_book_ids)]
        if start and end:
            df = df[(df.index.get_level_values('date') >= start) & (df.index.get_level_values('date') <= end)].sort_index()
        elif start:
            df = df[(df.index.get_level_values('date') >= start)].sort_index()
        elif end:
            df = df[(df.index.get_level_values('date') <= end)].sort_index()
        return df
    
    def get_index_xmin_consistent_prices(self, order_book_ids:list, start=None, end=None):
        """
        获取股指期货1分钟级别连续行情
        Args:
            order_book_ids: 股指合约过滤列表
            start:起始时间,默认为None,不进行过滤
            end:结束时间,默认为None,不进行过滤

        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.future_xmin_consistent_prices_filepath)
        df = pd.read_feather(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index(['order_book_id', 'datetime'], inplace=True)
        # print(df)
        if order_book_ids:
            df = df[df.index.get_level_values('order_book_id').isin(order_book_ids)]
        if start and end:
            df = df[(df.index.get_level_values('datetime') >= start) & (df.index.get_level_values('datetime') <= end)].sort_index()
        elif start:
            df = df[(df.index.get_level_values('datetime') >= start)].sort_index()
        elif end:
            df = df[(df.index.get_level_values('datetime') <= end)].sort_index()
        return df

    def get_simple_factor_1d_data(self, factors:list, order_book_ids:list, start=None, end=None, parallel=True)->pd.DataFrame:
        """
        获取日度简单因子数据
        Args:
            factors: 因子列表
            order_book_ids:标的的list, None则不进行人任何过滤
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤
            parallel:是否支持并行,默认为True
        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if not parallel:
            df_list = list()
            for factor_name in factors:
                filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}.feather")
                data = pd.read_feather(filepath)
                data['date'] = pd.to_datetime(data['date'])
                data['factor'] = factor_name
                data.set_index(['factor', 'date'], inplace=True)
                data.index.names = ['factor', 'date']
                if start and end:
                    data = data[(data.index.get_level_values('date') >= start) & (data.index.get_level_values('date') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('date') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('date') <= end)].sort_index()
                else:
                    data = data.sort_index()
                df_list.append(data)
            df = pd.concat(df_list)
        else:
            def _get_simple_factor_1d_data(factor_name, start, end):
                filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}.feather")
                data = pd.read_feather(filepath)
                data['date'] = pd.to_datetime(data['date'])
                data['factor'] = factor_name
                data.set_index(['factor', 'date'], inplace=True)
                data.index.names = ['factor', 'date']
                if start and end:
                    data = data[(data.index.get_level_values('date') >= start) & (data.index.get_level_values('date') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('date') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('date') <= end)].sort_index()
                else:
                    data = data.sort_index()
                return data
            
            jobs = len(factors)
            if jobs > 128:
                jobs = 128
            df_list = Parallel(n_jobs=jobs)(delayed(_get_simple_factor_1d_data)(factor_name, start, end) for factor_name in factors)
            df = pd.concat(df_list)
        if order_book_ids:
            df = df[order_book_ids]
        return df
    
    def get_compound_factor_1d_data(self, factors:list, order_book_ids:list, benchmark=None, start=None, end=None, parallel=True)->pd.DataFrame:
        """
        获取日度复合因子数据
        Args:
            factors: 因子列表
            order_book_ids:标的的list, None则不进行人任何过滤
            benchmark:基准指数
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤
            parallel:是否支持并行,默认为True
        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if not parallel:
            df_list = list()
            for factor_name in factors:
                if benchmark is not None:
                    filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}_{benchmark}.feather")
                else:
                    filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}.feather")
                data = pd.read_feather(filepath)
                data['date'] = pd.to_datetime(data['date'])
                data['factor'] = factor_name
                data.set_index(['factor', 'date'], inplace=True)
                data.index.names = ['factor', 'date']
                if start and end:
                    data = data[(data.index.get_level_values('date') >= start) & (data.index.get_level_values('date') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('date') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('date') <= end)].sort_index()
                else:
                    data = data.sort_index()
                df_list.append(data)
            df = pd.concat(df_list)
        else:
            def _get_compound_factor_1d_data(factor_name, start, end):
                if benchmark is not None:
                    filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}_{benchmark}.feather")
                else:
                    filepath = os.path.join(self.data_path, self.factor_post_1d_datapath, f"{factor_name}.feather")
                data = pd.read_feather(filepath)
                data['date'] = pd.to_datetime(data['date'])
                data['factor'] = factor_name
                data.set_index(['factor', 'date'], inplace=True)
                data.index.names = ['factor', 'date']
                if start and end:
                    data = data[(data.index.get_level_values('date') >= start) & (data.index.get_level_values('date') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('date') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('date') <= end)].sort_index()
                else:
                    data = data.sort_index()
                return data
            
            jobs = len(factors)
            if jobs > 128:
                jobs = 128
            df_list = Parallel(n_jobs=jobs)(delayed(_get_compound_factor_1d_data)(factor_name, start, end) for factor_name in factors)
            df = pd.concat(df_list)
        if order_book_ids:
            df = df[order_book_ids]
        return df
    
    def get_compound_factor_xmin_data(self, factors:list, order_book_ids:list, frequency:str, benchmark='sp1000', start=None, end=None, parallel=True)->pd.DataFrame:
        """
        获取分钟级别复合因子数据
        Args:
            factors: 因子列表
            order_book_ids:标的的list, None则不进行人任何过滤
            frequency:频率, 5min,15min,30min
            benchmark:基准指数,默认sp1000
            start:起始时间,默认为None,不进行过滤
            end:结束时间,默认为None,不进行过滤
            parallel:是否支持并行,默认为True
        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        factor_post_xmin_datapath = self.factor_post_xmin_datapath.format(frequency=frequency)
        if not parallel:
            df_list = list()
            for factor_name in factors:
                filepath = os.path.join(self.data_path, factor_post_xmin_datapath, f"{factor_name}_{benchmark}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data['factor'] = factor_name
                data.set_index(['factor', 'datetime'], inplace=True)
                data.index.names = ['factor', 'datetime']
                if start and end:
                    data = data[(data.index.get_level_values('datetime') >= start) & (data.index.get_level_values('datetime') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('datetime') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('datetime') <= end)].sort_index()
                else:
                    data = data.set_index(['factor', 'datetime']).sort_index()
                df_list.append(data)
            df = pd.concat(df_list)
        else:
            def _get_compound_factor_xmin_data(factor_name, start, end):
                filepath = os.path.join(self.data_path, factor_post_xmin_datapath, f"{factor_name}_{benchmark}.feather")
                data = pd.read_feather(filepath)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data['factor'] = factor_name
                data.set_index(['factor', 'datetime'], inplace=True)
                data.index.names = ['factor', 'datetime']
                if start and end:
                    data = data[(data.index.get_level_values('datetime') >= start) & (data.index.get_level_values('datetime') <= end)].sort_index()
                elif start:
                    data = data[(data.index.get_level_values('datetime') >= start)].sort_index()
                elif end:
                    data = data[(data.index.get_level_values('datetime') <= end)].sort_index()
                else:
                    data = data.sort_index()
                return data
            
            jobs = len(factors)
            if jobs > 128:
                jobs = 128
            df_list = Parallel(n_jobs=jobs)(delayed(_get_compound_factor_xmin_data)(factor_name, start, end) for factor_name in factors)
            df = pd.concat(df_list)
        if order_book_ids:
            df = df[order_book_ids]
        return df
    
    def get_rqfactor_1d_data(self, factor_name:str, start=None, end=None)->pd.DataFrame:
        """
        获取日度米筐因子数据
        Args:
            factor_name: 因子名称
            start:起始日期,默认为None,不进行过滤
            end:结束日期,默认为None,不进行过滤
        Returns:
            pd.DataFrame: 返回类型
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        filepath = os.path.join(self.data_path, self.rqfactor_post_1d_datapath, f"{factor_name}.feather")
        data = pd.read_feather(filepath)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index(['date'], inplace=True)
        data.index.names = ['date']
        if start is not None and end is not None:
            data = data[(data.index.get_level_values('date') >= start) & (data.index.get_level_values('date') <= end)].sort_index()
        elif start is not None:
            data = data[(data.index.get_level_values('date') >= start)].sort_index()
        elif end is not None:
            data = data[(data.index.get_level_values('date') <= end)].sort_index()
        else:
            data = data.sort_index()

        data = pd.DataFrame(data.stack()) 
        data.index.names=['date', 'order_book_id']
        data.columns=[f'{factor_name}']
        data.reset_index(inplace=True)
        data.set_index(['date', 'order_book_id'], inplace=True)
        return data
    
    def get_all_stock_list(self):
        df_basics = self.get_stock_basic_info()
        return df_basics.index.tolist()
    
    def get_all_trading_dates(self):
        """获取本地储存的所有交易日序列"""
        TRADING_CALENDAR_FILEPATH = os.path.join(self.data_path, f'stocks/basics/calendar.pickle')
        return [pd.to_datetime(dt) for dt in pd.read_pickle(TRADING_CALENDAR_FILEPATH)]

    def get_trading_dates(self, start_date, end_date, inclusive='both'):
        """获取交易日序列，左闭右闭"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        assert end_date > start_date
        _all_dates = self.get_all_trading_dates()
        _s = 0
        _e = -1
        for i, dt in enumerate(_all_dates):
            if (dt >= start_date) and (_s == 0):
                _s = i
            if (dt >= end_date) and (_e == -1):
                _e = i
                break
        return _all_dates[_s:_e + 1]

    def get_latest_trading_date(self, future=True):
        """获取最近的交易日（不含当日）"""
        _all_trddt = get_all_trading_dates()
        _today = datetime.datetime.today()
        _yesterday = _today - datetime.timedelta(days=1)
        _tomorrow = _today + datetime.timedelta(days=1)
        if future:
            return self.get_next_trading_date(_yesterday)
        else:
            return self.get_previous_trading_date(_tomorrow)

    def get_previous_trading_date(self, dt, n=1):
        """返回给定日期前N日的交易日"""
        _dt = pd.to_datetime(dt)
        _all_trddt = self.get_all_trading_dates()

        _n = 0
        for trddt in _all_trddt[::-1]:
            if trddt < _dt:
                _n += 1
            if _n >= n:
                return trddt
        print("无更小日期，返回原日期", _dt)
        return _dt

    def get_next_trading_date(self, dt, n=1):
        """
        获取下一个交易日
        - 每次直接获取本地所有交易日,然后获取给定交易日下一个交易日
        """
        _dt = pd.to_datetime(dt)
        _all_trddt = self.get_all_trading_dates()

        _n = 0
        for trddt in _all_trddt:
            if trddt > _dt:
                _n += 1
            if _n >= n:
                return trddt
        print("无更大日期，返回原日期", _dt)
        return _dt

    def get_index_futures_basis(self, order_book_ids:list, start_date=None, end_date=None, price_label='close', annual=True, **kwargs):
        
        def get_spot_order_book_ids(futures_order_book_ids):
            import re
            """给定期货合约id，返回现货id"""
            dict_futures_spot = {
                'IM': '000852.XSHG',  # 中证1000
                'IC': '000905.XSHG',  # 中证500
                'IH': '000050.XSHG',  # 上证50
                "IF": '000300.XSHG',  # 沪深300
            }
            spot_order_book_ids = set()
            dict_spot_futures = dict()
            _f_order_book_ids = list(sorted(futures_order_book_ids))
            for f_order_book_id in _f_order_book_ids:
                _f_type = re.findall("I[MCHF]", f_order_book_id)
                if len(_f_type) > 0:
                    s_order_book_id = dict_futures_spot[_f_type[0]]
                    dict_spot_futures[f_order_book_id] = s_order_book_id
                    spot_order_book_ids.add(s_order_book_id)
            return list(spot_order_book_ids), dict_spot_futures

        uniq_idx = set([obid.split('_')[0] for obid in order_book_ids])
        _df_futures_prices = self.get_index_consistent_prices(order_book_ids, start_date, end_date)
        _spot_order_book_ids, _dict_spot_futures = get_spot_order_book_ids(order_book_ids)
        _df_spot_prices = self.get_stock_post_1d_data(_spot_order_book_ids, start_date, end_date)

        _df_f_prices_p = _df_futures_prices[price_label].unstack().T
        _df_s_prices_p = _df_spot_prices[price_label].unstack().T
        # 循环编写基差计算
        list_basis = []
        for futures, spot in _dict_spot_futures.items():
            _s_f_basis = (_df_f_prices_p[futures] - _df_s_prices_p[spot]) / _df_s_prices_p[spot]
            _s_f_basis.name = futures
            list_basis.append(_s_f_basis)

        _df_s_f_basis = pd.concat(list_basis, axis=1)
        if annual:
            def get_index_futures_days_to_maturity(idx, start_date=None, end_date=None, no_zero=True):
                _df_mtu = self.get_future_maturity_date_data([idx], start_date, end_date)
                # 默认本地储存的日期非datetime格式
                _df_mtu = _df_mtu.apply(pd.to_datetime)
                _df_days_to_mtu = (
                    _df_mtu.sub(_df_mtu.index, axis=0)
                    .apply(lambda i: i.dt.days)
                    # .replace(0, np.NaN)  # 目前使用的是收盘价，交割日不计算
                )
                if no_zero:
                    _df_days_to_mtu.replace(0, np.NaN, inplace=True)
                return _df_days_to_mtu
    
            _df_days_to_mtu = get_index_futures_days_to_maturity(uniq_idx.pop(), start_date, end_date)
            _df_s_f_basis = _df_s_f_basis.mul((365 / _df_days_to_mtu))
        return _df_s_f_basis
    
    
if __name__ == "__main__":
    # data_source_path = r"\\192.168.1.168\samba\Market"
    data_source_path = r"\\192.168.3.100\samba\Market"
    client = StockDataClient(data_path=data_source_path)
    
    # 基础信息
    print("基础信息")
    df = client.get_stock_basic_info()
    print(df)
    
    # 股票ST信息
    print("股票ST信息")
    df = client.get_stock_st_info()
    print(df)
    
    order_book_ids = ['000001.XSHE', '000004.XSHE', '688800.XSHG']
    df = client.get_stock_st_info(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_st_info(order_book_ids=order_book_ids, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 股票停牌信息
    print("股票停牌信息")
    df = client.get_stock_suspended_info()
    print(df)
    
    order_book_ids = ['000001.XSHE', '000004.XSHE', '688800.XSHG']
    df = client.get_stock_suspended_info(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_suspended_info(order_book_ids=order_book_ids, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 指数成分
    print("指数成分")
    df = client.get_stock_index_comments(order_book_id='000016.XSHG')
    print(df)
    
    df = client.get_stock_index_comments(order_book_id='000016.XSHG', start='2010-01-01', end='2019-12-31')
    print(df)

    # 指数成分
    print("指数成分")
    df = client.get_stock_index_comments(order_book_id='000510.XSHG')
    print(df)
    
    # 指数权重
    print("指数权重")
    df = client.get_stock_index_comments_weights(order_book_id='000016.XSHG')
    print(df)
    
    df = client.get_stock_index_comments_weights(order_book_id='000016.XSHG', start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 股票日度行情
    print("股票日度行情")
    order_book_ids = ['000001.XSHE']  
    df = client.get_stock_post_1d_data(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_post_1d_data(order_book_ids=order_book_ids, start='2010-01-01')
    print(df)
    
    df = client.get_stock_post_1d_data(order_book_ids=order_book_ids, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 股票日度VWAP行情
    print("股票日度VWAP行情")
    order_book_ids = ['000001.XSHE']
    df = client.get_stock_post_vwap_1d_data(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_post_vwap_1d_data(order_book_ids=order_book_ids, start='2010-01-01')
    print(df)
    
    df = client.get_stock_post_vwap_1d_data(order_book_ids=order_book_ids, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 股票日度TWAP行情
    print("股票日度TWAP行情")
    order_book_ids = ['000001.XSHE']
    df = client.get_stock_post_twap_1d_data(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_post_twap_1d_data(order_book_ids=order_book_ids, start='2010-01-01')
    print(df)
    
    df = client.get_stock_post_twap_1d_data(order_book_ids=order_book_ids, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 股票1min行情
    print("股票1min行情")
    order_book_ids = ['000001.XSHE']
    df = client.get_stock_post_1min_data(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_stock_post_1min_data(order_book_ids=order_book_ids, start='2010-01-01 00:00:00')
    print(df)
    
    df = client.get_stock_post_1min_data(order_book_ids=order_book_ids, start='2010-01-01 00:00:00', end='2019-12-31 15:30:00')
    print(df)
    
    # 股票5minVWAP/TWAP行情
    print("股票5minVWAP/TWAP行情")
    order_book_ids = ['000001.XSHE']
    df = client.get_stock_post_vwap_twap_xmin_data(frequency='5min', order_book_ids=order_book_ids, parallel=False)
    print(df)
    
    df = client.get_stock_post_vwap_twap_xmin_data(frequency='15min', order_book_ids=order_book_ids, start='2010-01-01 00:00:00')
    print(df)
    
    df = client.get_stock_post_vwap_twap_xmin_data(frequency='30min', order_book_ids=order_book_ids, start='2010-01-01 00:00:00', end='2019-12-31 15:30:00')
    print(df)
    
    # 股指1min行情
    print("股指1min行情")
    order_book_ids = ['IH2410', 'IF2410', 'IC2410', 'IM2410']
    df = client.get_future_post_1min_data(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_future_post_1min_data(order_book_ids=order_book_ids, start='2024-09-01 00:00:00')
    print(df)
    
    df = client.get_future_post_1min_data(order_book_ids=order_book_ids, start='2024-09-01 00:00:00', end='2024-12-31 15:30:00')
    print(df)
    
    # 股指期货合约状态
    print("股指期货合约状态")
    products = ['IH', 'IF', 'IC', 'IM']
    df = client.get_future_state_matrix_data(products=products)
    print(df)
    
    df = client.get_future_state_matrix_data(products=products, start='2024-09-01')
    print(df)
    
    df = client.get_future_state_matrix_data(products=products, start='2024-09-01', end='2024-12-31')
    print(df)
    
    # 股指期货合约到期日
    print("股指期货合约到期日")
    products = ['IH', 'IF', 'IC', 'IM']
    df = client.get_future_maturity_date_data(products=products)
    print(df)
    
    df = client.get_future_maturity_date_data(products=products, start='2010-01-01')
    print(df)
    
    df = client.get_future_maturity_date_data(products=products, start='2010-01-01', end='2024-12-31')
    print(df)
    
    # 股指连续价格
    print("股指连续价格")
    order_book_ids = ['IH_CM', 'IF_NM', 'IC_CQ', 'IM_NQ']
    df = client.get_index_consistent_prices(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_index_consistent_prices(order_book_ids=order_book_ids, start='2024-09-01')
    print(df)
    
    df = client.get_index_consistent_prices(order_book_ids=order_book_ids, start='2024-09-01', end='2024-12-31')
    print(df)
    
    # 股指1min连续价格
    print("股指1min连续价格")
    order_book_ids = ['IH_CM', 'IF_NM', 'IC_CQ', 'IM_NQ']
    df = client.get_index_xmin_consistent_prices(order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_index_xmin_consistent_prices(order_book_ids=order_book_ids, start='2024-09-01 00:00:00')
    print(df)
    
    df = client.get_index_xmin_consistent_prices(order_book_ids=order_book_ids, start='2024-09-01 00:00:00', end='2024-12-31 15:30:00')
    print(df)
    
    order_book_ids = ['603826.XSHG', '600753.XSHG', '300231.XSHE', '600200.XSHG', '300625.XSHE']
    # 日度简单因子数据
    print("日度简单因子数据")
    factors = ['mor_corr_volume_ret', 'corr_volume_ntrades', 'corr_close_nextopen']
    df = client.get_simple_factor_1d_data(factors=factors, order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_simple_factor_1d_data(factors=factors, order_book_ids=order_book_ids, start='2010-01-01')
    print(df)
    
    df = client.get_simple_factor_1d_data(factors=factors, order_book_ids=None, start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 日度复合因子数据
    print("日度复合因子数据 benchmark=")
    factors = ['tobt240', 'tobt480', 'tobt720']
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=order_book_ids, benchmark='000852.XSHG')
    print(df)
    
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=order_book_ids, benchmark='000852.XSHG', start='2010-01-01')
    print(df)
    
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=None, benchmark='000852.XSHG', start='2010-01-01', end='2019-12-31')
    print(df)
    
    # 日度复合因子数据
    print("日度复合因子数据")
    factors = ['tobt240', 'tobt480', 'tobt720']
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=order_book_ids)
    print(df)
    
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=order_book_ids, start='2010-01-01')
    print(df)
    
    df = client.get_compound_factor_1d_data(factors=factors, order_book_ids=None, start='2010-01-01', end='2019-12-31')
    print(df)
    
    
    # 5min级别符合因子数据
    print("5min级别符合因子数据")
    factors = ['roc96', 'roc144', 'roc240']
    df = client.get_compound_factor_xmin_data(factors=factors, order_book_ids=order_book_ids, frequency='5min', benchmark='sp1000')
    print(df)
    
    df = client.get_compound_factor_xmin_data(factors=factors, order_book_ids=order_book_ids, frequency='5min', benchmark='sp1000', start='2010-01-01 00:00:00')
    print(df)
    
    df = client.get_compound_factor_xmin_data(factors=factors, order_book_ids=None, frequency='5min', benchmark='sp1000', start='2010-01-01 00:00:00', end='2019-12-31 15:30:00')
    print(df)

    
    # 米筐因子数据
    print("米筐因子数据")
    df = client.get_rqfactor_1d_data(factor_name='total_turnover')
    print(df)
    
    df = client.get_rqfactor_1d_data(factor_name='total_turnover', start='2010-01-01')
    print(df)
    
    df = client.get_rqfactor_1d_data(factor_name='total_turnover', start='2010-01-01', end='2019-12-31')
    print(df)
    