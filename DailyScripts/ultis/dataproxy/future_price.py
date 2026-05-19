import os
import traceback
import pandas as pd
import time
from datetime import datetime
from joblib import Parallel, delayed
from loguru import logger # type: ignore
import rqdatac # type: ignore


def update_futures_minute_prices(data_path, latest=True):
    """
    更新期货分钟级数据
        - 目前更新程序里只更新特定股指期货的数据
        - 新上市的股票数据在另外的脚本里统一本地新生成
        - 米筐期货行情数据在每日的16:30入库,但测试结果中股指期货的数据是实时更新,16点30再更新(股票数据更新后更新)
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logger.info("启动期货分钟级行情数据更新")
    now = datetime.now()
    if now.hour >= 16:
        update_end_trddt = rqdatac.get_latest_trading_date()
    else:
        update_end_trddt = rqdatac.get_previous_trading_date(now)
    if not latest:
        update_end_trddt = rqdatac.get_previous_trading_date(rqdatac.get_latest_trading_date())
    target_idx_futures = ['IM', 'IC', 'IF', 'IH']
    all_futures = rqdatac.all_instruments(type='Future', market='cn', date=update_end_trddt)
    all_idx_futures = all_futures[
        (all_futures['order_book_id'].str.slice(0, 2).isin(target_idx_futures))
        & (all_futures['order_book_id'].str[2].isin(["1", "2"]))
    ].set_index('order_book_id')

    def _download_latest_data(_order_book_id):
        _fp = os.path.join(data_path, f'{_order_book_id}.feather')
        if not os.path.exists(_fp):
            _start_date = all_idx_futures.loc[_order_book_id]['listed_date']
            update_start_trddt = pd.to_datetime(_start_date)
        else:
            _df_local = pd.read_feather(_fp)
            _df_local.set_index(['datetime'], inplace=True)
            update_start_trddt = rqdatac.get_next_trading_date(_df_local.index.max().date())
        if update_start_trddt > update_end_trddt:
            logger.info(f"{_order_book_id}当前已是最新数据，跳过 {update_start_trddt} to {update_end_trddt}")
            return None, None

        logger.info(f"正在下载{_order_book_id}数据，{update_start_trddt} to {update_end_trddt}")
        _df_min_prices = rqdatac.get_price(_order_book_id, start_date=update_start_trddt, end_date=update_end_trddt, expect_df=False, frequency='1m')
        if _df_min_prices is None:
            # None一般是针对已到期的期货，在更新时会出现_start_date > end_date的情形导致None return
            logger.warning(f"没有获取到{_order_book_id}分钟行情数据，时间范围:{update_start_trddt} to {update_end_trddt}")
            return None, None
        # FIXMEd：因为米筐返回的股指数据好像是实时的，为了避免重复插入这里限制终点
        # _df_min_prices = _df_min_prices.loc[:end_date]
        if not _df_min_prices.empty:
            if not os.path.exists(_fp):
                df_all = _df_min_prices
            else:
                df_all = pd.concat([_df_local, _df_min_prices]).sort_index()
            df_all.reset_index(inplace=True)
            start_time = time.time()
            df_all.to_feather(_fp)
            end_time = time.time()
            logger.info(f"完成{_order_book_id}分钟级别数据更新, {_fp}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all)/(end_time-start_time), 2)}row/s")
        else:
            logger.info(f"{_order_book_id}没有新的数据需要更新")
        return _fp, _df_min_prices
    
    results = list()
    for order_book_id in all_idx_futures.index.unique().tolist():
        result = _download_latest_data(order_book_id)
        results.append(result)
    logger.info("期货分钟级行情数据更新完成")
    return results


def update_futures_state_matrix(index_state_path, index_maturity_path):
    logger.info("启动期货合约状态和到期日数据更新")
    result = dict()
    if not os.path.exists(os.path.dirname(index_state_path)):
        os.makedirs(os.path.dirname(index_state_path))
    if not os.path.exists(os.path.dirname(index_maturity_path)):
        os.makedirs(os.path.dirname(index_maturity_path))
    def _get_future_state_matrix(idx_future, start_date, end_date):
        assert idx_future in ['IM', 'IC', 'IH', 'IF']
        all_trddts = rqdatac.get_trading_dates(start_date, end_date)
        list_s = []
        list_s_mtu = []
        _futures_cols = [f'{idx_future}_CM', f'{idx_future}_NM', f'{idx_future}_CQ', f'{idx_future}_NQ', ]
        for _trddt in all_trddts:
            _futures = rqdatac.all_instruments(type='Future', market='cn', date=_trddt)
            _idx_futures = _futures[(_futures['order_book_id'].str.slice(0, 2).isin([idx_future, ])
                                     & (_futures['maturity_date'] != '0000-00-00')
                                     )].sort_values(by=['order_book_id'])[['order_book_id', 'maturity_date']].reset_index(drop=True)
            _idx_futures['column'] = _futures_cols
            _s = _idx_futures[['order_book_id', 'column']].set_index('column')['order_book_id']
            _s.name = _trddt.strftime("%Y-%m-%d")
            _s_mtu = _idx_futures[['maturity_date', 'column']].set_index('column')['maturity_date']
            _s_mtu.name = _trddt.strftime("%Y-%m-%d")
            list_s.append(_s)
            list_s_mtu.append(_s_mtu)
        _df = pd.concat(list_s, axis=1).T
        _df.index.name = 'date'

        _df_mtu = pd.concat(list_s_mtu, axis=1).T
        _df_mtu.index.name = 'date'
        return _df, _df_mtu

    DICT_FUTURES_LISTED_DATE = {
        "IM": "2022-07-22",
        "IC": "2015-04-16",
        "IH": "2015-04-16",
        "IF": "2010-04-16",
    }
    update_end_trddt = rqdatac.get_latest_trading_date()
    product_state_list = list()
    product_maturity_list = list()
    
    df_local_state_matrix = pd.read_feather(index_state_path)
    df_local_state_matrix.set_index(['date'], inplace=True)
    df_local_mtu_dt = pd.read_feather(index_maturity_path)
    df_local_mtu_dt.set_index(['date'], inplace=True)
    local_max_dt = pd.to_datetime(df_local_state_matrix.index.max(), format='%Y-%m-%d')
    now = datetime.now()
    if now.hour >= 16:
        update_end_trddt = rqdatac.get_latest_trading_date()
    else:
        update_end_trddt = rqdatac.get_previous_trading_date(now)
    update_start_trddt = rqdatac.get_next_trading_date(local_max_dt)
    if update_start_trddt > update_end_trddt:
        logger.info(f"本地已经是最新状态数据 跳过，{update_start_trddt} to {update_end_trddt}")
        return result

    for idx, st_dt in DICT_FUTURES_LISTED_DATE.items():
        logger.info(f"更新{idx}期货合约状态和到期日数据 时间范围：{update_start_trddt} to {update_end_trddt}")
        df_new_state_matrix, df_new_mtu_dt = _get_future_state_matrix(idx, st_dt, update_end_trddt)
        df_new_state_matrix['product'] = idx
        df_new_mtu_dt['product'] = idx
        product_state_list.append(df_new_state_matrix)
        product_maturity_list.append(df_new_mtu_dt)
        result[idx] = [(index_state_path, df_new_state_matrix), (index_maturity_path, df_new_mtu_dt)]
        
    df_state_matrix_all = pd.concat(product_state_list)
    df_state_matrix_all.reset_index(inplace=True)
    df_maturity_date_all = pd.concat(product_maturity_list)
    df_maturity_date_all.reset_index(inplace=True)
    start_time = time.time()
    df_state_matrix_all.to_feather(index_state_path)
    df_maturity_date_all.to_feather(index_maturity_path)
    end_time = time.time()
    logger.info(f"完成期货合约的状态和到期日数据更新, {index_state_path}和{index_maturity_path}写入完成, 耗时：{round(end_time-start_time, 2)}s 时间范围:{st_dt} to {update_end_trddt}")
    return result


def update_futures_consistent_prices(data_1min_path, data_1d_path, index_state_path, frequency='1d'):
    """
    股指期货当月、次月、远月、次季数据合成更新
    """
    if not os.path.exists(data_1min_path):
        os.makedirs(data_1min_path)
    if not os.path.exists(data_1d_path):
        os.makedirs(data_1d_path)
    logger.info(f"启动期货{frequency}级别行情拼接数据更新(当月、次月、远月、次季)")
    end_date = rqdatac.get_latest_trading_date()
    target_idx_futures = ['IM', 'IC', 'IF', 'IH']
    futures_types = ['CM', 'NM', 'CQ', "NQ"]
    def _get_all_idx_future_prices(idx_future):
        _files = [file for file in os.listdir(data_1min_path) if (idx_future in file) and (file[3:5] not in futures_types)]
        list_df = []
        for _file in _files:
            _order_book_id = _file.split('.')[0]
            _fp = os.path.join(data_1min_path, _file)
            _df = pd.read_feather(_fp)
            _df.set_index(['datetime'], inplace=True)
            _df['order_book_id'] = _order_book_id
            list_df.append(_df.reset_index())
        _df_all = pd.concat(list_df)
        return _df_all

    def _twap(i):
        return i.mean()

    def _vwap(df):
        _df = df.copy(deep=True)
        _df['close'] = _df['close'] * _df['volume']
        df_d = _df.groupby(_df.index.date)[['close', 'volume']].sum()
        s_vwap = df_d['close'] / df_d['volume']
        s_vwap.name = 'vwap'
        s_vwap.index.name = 'date'
        return s_vwap

    def _vwap_last1hour(df):
        """计算最后1H的vwap"""
        _df = df.between_time('14:01', '15:01')
        return _vwap(_df)

    def _get_daily_future_prices(df_min_future_prices, order_book_id):
        df_daily_futures_prices = df_min_future_prices.groupby(df_min_future_prices.index.date).agg({
            'open': 'first',
            'high': 'max', 'low': 'min',
            'close': ['last', _twap],
            'volume': 'sum', 'total_turnover': 'sum', 'open_interest': 'last',
        })
        s_vwap = _vwap(df_min_future_prices)
        df_daily_futures_prices['vwap'] = s_vwap
        df_daily_futures_prices.reset_index(inplace=True)
        df_daily_futures_prices.columns = ['date', 'open', 'high', 'low', 'close', 'twap',
                                           'volume', 'total_turnover', 'open_interest',
                                           'vwap']
        df_daily_futures_prices['date'] = pd.to_datetime(df_daily_futures_prices['date'])
        df_daily_futures_prices['order_book_id'] = order_book_id
        df_daily_futures_prices.set_index(['order_book_id', 'date'], inplace=True)
        df_daily_futures_prices.sort_index(inplace=True)
        return df_daily_futures_prices
    
    def _get_index_futures_state_matrix(index_state_path, idx, start_date=None, end_date=None):
        """
        给定指数期货代码,返回连续的各期期货状态矩阵
        """
        _df = pd.read_feather(index_state_path)
        if start_date is not None:
            _df = _df.loc[start_date:]
        if end_date is not None:
            _df = _df.loc[:end_date]
        return _df
    
    def _get_futures_consistent_prices(idx_future):
        """
        给定指数期货类型(例如IM),返回各周期期货的连续行情数据
        :return: 连续的行情数据，以(order_book_id, date)为复合索引,以close, twap, open, vwap等为列
        """
        df_futures_state_matrix = _get_index_futures_state_matrix(index_state_path=index_state_path, idx=idx_future)
        df_futures_state_matrix.set_index(['date'], inplace=True)
        df_futures_state_matrix.index = pd.to_datetime(df_futures_state_matrix.index)
        df_all_prices = _get_all_idx_future_prices(idx_future)
        df_all_prices['date'] = pd.to_datetime(df_all_prices['datetime'].dt.date)
        list_daily_prices = []
        futures_types = ['CM', 'NM', 'CQ', "NQ"]  # 四种期货类型：当月、次月、当季、次季
        for idx_future_type in futures_types:
            idx_future_order_book_id = f"{idx_future}_{idx_future_type}"
            # print(df_futures_state_matrix[[idx_future_order_book_id]].reset_index().rename(columns={idx_future_order_book_id: 'order_book_id'}))
            df_idx_future_prices = df_all_prices.merge(
                df_futures_state_matrix[[idx_future_order_book_id]].reset_index().rename(columns={idx_future_order_book_id: 'order_book_id'}),
                on=['date', 'order_book_id'], how='inner').drop(columns=['date'])
            if frequency == '1d':
                df_idx_future_prices = _get_daily_future_prices(df_idx_future_prices.set_index('datetime'), idx_future_order_book_id)
                # df_idx_future_prices_daily.head()
            else:
                df_idx_future_prices['order_book_id'] = idx_future_order_book_id
                df_idx_future_prices.set_index(['order_book_id', 'datetime'], inplace=True)

            list_daily_prices.append(df_idx_future_prices)
        df_idx_futures_daily = pd.concat(list_daily_prices)
        return df_idx_futures_daily

    dirpath = {
        '1d': data_1d_path,
        '1min': data_1min_path,
    }.get(frequency)

    list_futures_consistent_prices = []
    for index_future in target_idx_futures:
        df_index_futures_consistent_prices = _get_futures_consistent_prices(index_future)
        list_futures_consistent_prices.append(df_index_futures_consistent_prices)
        logger.info(f"完成{index_future}合约数据加载")
    # if not split_save:
    df_all_futures_consistent_prices = pd.concat(list_futures_consistent_prices)
    if frequency == '1min':
        fp_all = f'{dirpath}/index_futures_xmin_consistent_prices.feather'
    elif frequency == '1d':
        fp_all = f'{dirpath}/index_futures_daily_consistent_prices.feather'
    
    df_all_futures_consistent_prices.reset_index(inplace=True)
    start_time = time.time()
    df_all_futures_consistent_prices.to_feather(fp_all)
    end_time = time.time()
    logger.info(f"{frequency}持续合约行情数更新完成, {fp_all}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_futures_consistent_prices)/(end_time-start_time), 2)}row/s")
    return fp_all, df_all_futures_consistent_prices
