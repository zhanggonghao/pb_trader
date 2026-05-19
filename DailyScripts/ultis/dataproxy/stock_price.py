"""
负责数据的本地化和更新

# TODOd：定时更新市场基本数据
#  细分1：个股列表
#  细分2：交易日
#  细分3：停牌、ST统计
#  细分4：指数及指数成分股数
# TODOd：周度更新分钟级数据
#  细分1：个股分钟级(指数分钟级)
"""

import os
import traceback
import pandas as pd
import time
from datetime import datetime
from joblib import Parallel, delayed
from loguru import logger # type: ignore
import rqdatac # type: ignore
        

def update_stock_minute_prices_parallel(data_1min_path, jobs=8, latest=True):
    result = dict()
    if not os.path.exists(data_1min_path):
        os.makedirs(data_1min_path)
    logger.info(f"启动个股分钟级行情数据(后复权)更新 并行数量：{jobs}")
    adjust_type = 'post'
    # 从本地目录文件提取所有股票
    all_local_stocks = [filename.split('.feather')[0] for filename in os.listdir(data_1min_path) if filename.endswith('.feather')]
    # 删除自己合成的sp1000
    if 'sp1000' in all_local_stocks:
        all_local_stocks.remove('sp1000')
    # 从平台获取所有股票
    df_all_stocks = rqdatac.all_instruments(type='CS', market='cn', date=None)
    # 如果本地是空目录，则使用平台获取的股票列表
    if len(all_local_stocks) == 0:
        all_local_stocks = df_all_stocks['order_book_id'].tolist()
    df_all_stocks = df_all_stocks[df_all_stocks['listed_date'] != '2999-12-31']  # 2999-12-31的并不是股票
    # 当前正在交易的股票
    all_listing_stocks = df_all_stocks[df_all_stocks['de_listed_date'] == '0000-00-00']['order_book_id'].tolist()  
    # 当前已退市的股票（部分可能处于无交易但尚未退市的退市整理期）
    all_delisted_stocks = df_all_stocks[df_all_stocks['de_listed_date'] != '0000-00-00']['order_book_id'].tolist()  
    # 本地+正在交易（因为本地数据中含指数数据，所以这里采用减法，即去掉本地退市的股票）
    local_listing_stocks = list(set(all_local_stocks).difference(set(all_delisted_stocks)))  
    # 新增股票数据
    new_listing_stocks = list(set(all_listing_stocks).difference(set(all_local_stocks)))  

    # 更新本地已包含的个股数据
    def _read_local_max_date(_order_book_id, data_1min_path):
        _fp = os.path.join(data_1min_path, f'{_order_book_id}.feather')
        if os.path.exists(_fp):
            # 读取最后100条记录
            try:
                _df = pd.read_feather(_fp)
                # print(_df)
                _df.set_index(['datetime'], inplace=True)
                _local_max_date = _df.index.max()
            except:
                raise KeyError(f"{_fp} open failed")
        else:
            _local_max_date = datetime(2000, 1, 4)
        return _order_book_id, _local_max_date
    
    local_max_dates = Parallel(n_jobs=jobs)(delayed(_read_local_max_date)(obid, data_1min_path) for obid in local_listing_stocks)
    df_local_max_dates = pd.DataFrame(local_max_dates, columns=['order_book_id', 'local_max_date'])
    uniq_local_max_date = df_local_max_dates['local_max_date'].unique()
    if len(uniq_local_max_date) > 1:
        logger.warning(f"本地股票数据的最新日期不同，需要分{len(uniq_local_max_date)}批次进行数据更新.")
    for local_max_date in uniq_local_max_date:
        ticker_list = df_local_max_dates[df_local_max_dates['local_max_date'] == local_max_date]['order_book_id'].to_list()
        ticker_list = list(set(ticker_list))
        # 更新起点：本地最新日期的下一个交易日
        update_start_trddt = rqdatac.get_next_trading_date(local_max_date.astype('datetime64[D]'))  
        # 更新终点：最新交易日的前一个交易日
        now = datetime.now()
        if now.hour >= 16:
            update_end_trddt = rqdatac.get_latest_trading_date()
        else:
            update_end_trddt = rqdatac.get_previous_trading_date(now)
        if not latest:
            update_end_trddt = rqdatac.get_previous_trading_date(update_end_trddt)
        if update_start_trddt > update_end_trddt:
            logger.info(f"{local_max_date}批次当前已是最新数据, 跳过 {update_start_trddt} to {update_end_trddt}")
            continue
        logger.info(f"{local_max_date}批次股票数量:{len(ticker_list)} 更新时间范围:{update_start_trddt} to {update_end_trddt}")
        # 增量获取最新行情数据
        df_local_listing_new_prices = rqdatac.get_price(ticker_list, update_start_trddt, 
                                                        update_end_trddt, frequency='1m', adjust_type=adjust_type)
        new_prices_order_book_ids = df_local_listing_new_prices.index.get_level_values('order_book_id').unique().tolist()
        logger.info(f"{local_max_date}批次{len(ticker_list)}只股票更新数据已下载完成，开始本地增量更新")

        def _local_update(_df_prices, _order_book_id, data_1min_path):
            _df_new_prices = _df_prices.loc[_order_book_id].dropna(axis=1, how='all')
            _fp = os.path.join(data_1min_path, f'{_order_book_id}.feather')
            if os.path.exists(_fp):
                df_local = pd.read_feather(_fp)
                df_local.set_index(['datetime'], inplace=True)
                df_all = pd.concat([df_local, _df_new_prices]).sort_index()
            else:
                df_all = _df_new_prices
            df_all.reset_index(inplace=True)
            df_all.to_feather(_fp)
            return _order_book_id, (_fp, _df_new_prices)
        
        if jobs > len(ticker_list):
            jobs = round(len(ticker_list) / 2) + 1
            
        results = Parallel(n_jobs=jobs)(delayed(_local_update)(df_local_listing_new_prices, obid, data_1min_path) for obid in new_prices_order_book_ids)
        for order_book_id, (filepath, df) in results:
            result[order_book_id] = (filepath, df)
        logger.info(f"{local_max_date}批次{len(ticker_list)}只股票本地数据增量更新完成")
    # 新股更新：新股一般不多，且上市时间不长，所以直接从最早的开始到最新数据（后续逐日更新后就是当天，第一次则不是）
    df_new_listing_stocks = df_all_stocks[(df_all_stocks['order_book_id'].isin(new_listing_stocks)) & 
                                          (df_all_stocks['listed_date'] <= datetime.now().strftime("%Y-%m-%d"))]
    new_listing_stocks = df_new_listing_stocks['order_book_id'].unique().tolist()
    if df_new_listing_stocks.empty:
        logger.info(f"{now:%Y-%m-%d}无新股上市，无新股数据需要更新")
        return result
    logger.info(f"{now:%Y-%m-%d}当日新股上市共{df_new_listing_stocks.shape[0]}只，开始更新新股数据")
    min_new_start_date = df_new_listing_stocks['listed_date'].min()
    if pd.to_datetime(min_new_start_date) <= update_end_trddt:
        df_new_listing_new_prices = rqdatac.get_price(
            new_listing_stocks, min_new_start_date, update_end_trddt,
            frequency='1m', adjust_type=adjust_type
        )
        for obid in new_listing_stocks:
            _df = df_new_listing_new_prices.loc[obid].dropna(axis=1, how='all')
            _fp = os.path.join(data_1min_path, f'{obid}.feather')
            _df.reset_index(inplace=True)
            start_time = time.time()
            _df.to_feather(_fp)
            end_time = time.time()
            logger.info(f"新股{obid}分钟级别数据更新完成, 时间范围:{_df.index.min()} to {_df.index.max()}, {_fp}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(_df)/(end_time-start_time), 2)}row/s")
            result[obid] = (_fp, _df)
    logger.info("完成个股分钟级行情数据(后复权)更新")
    return result


def update_stock_daily_prices(data_1d_filepath, latest=True):
    """
    个股日度数据更新（覆盖更新）
        - 由于日度整体数据量不大，所以本地每一次直接覆盖更新即可，后续复制至另外机器时通过脚本自动备份
    """
    if not os.path.exists(os.path.dirname(data_1d_filepath)):
        os.makedirs(os.path.dirname(data_1d_filepath))
    logger.info("启动个股日度行情数据(后复权)更新")
    adjust_type = 'post'
    if os.path.exists(data_1d_filepath):
        df_local = pd.read_feather(data_1d_filepath)
        df_local.set_index(['order_book_id', 'date'], inplace=True)
        code = df_local.index.to_list()[0][0]
        local_max_date = df_local.loc[code].index.max()
        update_start_trddt = rqdatac.get_next_trading_date(local_max_date)
    else:
        df_local = pd.DataFrame()
        local_max_date = datetime(2000, 1, 4)
        update_start_trddt = pd.to_datetime('2000-01-04')
        logger.warning(f"此前尚未生成VWAP数据在路径:{data_1d_filepath} 将从默认起点:{update_start_trddt}开始生成数据,耗时较久")
    # 更新终点定为：最新交易日的前一个交易日
    now = datetime.now()
    if now.hour >= 16:
        update_end_trddt = rqdatac.get_latest_trading_date()
    else:
        update_end_trddt = rqdatac.get_previous_trading_date(now)
    if not latest:
        update_end_trddt = rqdatac.get_previous_trading_date(now)
    if update_start_trddt > update_end_trddt:
        logger.info(f"{data_1d_filepath}已是最新数据，跳过 {update_start_trddt} to {update_end_trddt}")
        return data_1d_filepath, None
    
    logger.info(f"更新数据范围:{update_start_trddt} to {update_end_trddt}")
    all_stocks = rqdatac.all_instruments(type='CS', market='cn', date=None)['order_book_id'].tolist()
    idx_order_book_ids = [
        '000016.XSHG',  # 上证50
        '000300.XSHG',  # 沪深300
        '000688.XSHG',  # 科创50
        '399673.XSHE',  # 创业板50
        '000852.XSHG',  # 中证1000, 2014-10-17开始
        '000905.XSHG',  # 中证500
        '000906.XSHG',  # 中证800
        '399303.XSHE',  # 国证1000
        '399311.XSHE',  # 国证2000
        '000985.XSHG',  # 中证全指
        '000510.XSHG',
        # "000050.XSHG",  # 上证50（等权）
    ]
    all_stocks += idx_order_book_ids
    # 最新数据增量更新
    df_new = rqdatac.get_price(all_stocks, update_start_trddt, update_end_trddt, adjust_type=adjust_type)
    df_all = pd.concat([df_local, df_new]).sort_index()
    df_all.reset_index(inplace=True)
    start_time = time.time()
    df_all.to_feather(data_1d_filepath)
    end_time = time.time()
    logger.info(f"完成个股日度行情数据(后复权)全量更新 更新股票数量:{len(all_stocks)}, {data_1d_filepath}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all)/(end_time-start_time), 2)}row/s")
    return data_1d_filepath, df_new

def _calc_vwap(file_name, data_1min_path, update_start_date):
    update_start_date = pd.to_datetime(update_start_date)
    _fp = os.path.join(data_1min_path, file_name)
    # 增加时间约束，只计算更新部分的日度vwap
    _df = pd.read_feather(_fp)
    _df.set_index(['datetime'], inplace=True)
    try:
        _df = _df.loc[_df.index > update_start_date].reset_index()
    except:
        print(_df)
        raise KeyError(f'{update_start_date} {_fp}')
    # 这里这样提取date，意味着通过min数据计算的daily vwap数据的label=='left'，即T日的数据代表的是[T, T+1)的数据，closed=='left'
    _df['date'] = _df['datetime'].dt.strftime('%Y%m%d')
    _cols = ['open', 'high', 'low', 'close']
    _df[_cols] = _df[_cols].multiply(_df['volume'], axis=0)
    _df = _df.groupby(by=['date']).sum(numeric_only=True).reset_index()
    _df[_cols] = _df[_cols].div(_df['volume'], axis=0)
    _df.dropna(subset=_cols, inplace=True)

    _order_book_id = file_name.split('.feather')[0]
    _df['order_book_id'] = _order_book_id
    if 'datetime' in _df.columns:
        _df.drop(columns=['datetime'], inplace=True)
    _df.set_index(['order_book_id', 'date'], inplace=True)
    return _df

def update_stock_daily_vwap_prices(data_1min_path, vwap_path, jobs=8):
    """
    基于更新的分钟级数据增量更新本地VWAP数据
    """
    if not os.path.exists(os.path.dirname(data_1min_path)):
        os.makedirs(os.path.dirname(data_1min_path))
    if not os.path.exists(os.path.dirname(vwap_path)):
        os.makedirs(os.path.dirname(vwap_path))
    logger.info("启动个股日度VWAP数据(后复权)更新")
    adjust_type = 'post'
    # 首先读取本地储存的后复权VWAP价格数据
    if os.path.exists(vwap_path):
        df_local_stock_post_daily_vwap = pd.read_feather(vwap_path)
        df_local_stock_post_daily_vwap.set_index(['order_book_id', 'date'], inplace=True)
        code = df_local_stock_post_daily_vwap.index.to_list()[0][0]
        local_max_date = df_local_stock_post_daily_vwap.loc[code].index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)
    else:
        df_local_stock_post_daily_vwap = pd.DataFrame()
        update_start_date = pd.to_datetime('2000-01-04')
        logger.warning(f"此前尚未生成VWAP数据在路径:{vwap_path} 将从默认起点:{update_start_date}开始生成数据,耗时较久")
    """df_local_stock_post_daily_vwap.head(3)
                                 close         low  ...        open        high
    order_book_id date                              ...                        
    000001.XSHE   20050104  215.748292  215.502310  ...  215.746878  215.999912
                  20050105  213.062457  212.885621  ...  213.100181  213.271447
                  20050106  215.387209  215.191657  ...  215.377106  215.584354
    """
    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{vwap_path}当前已是最新数据，跳过 {update_start_date} to {update_end_date}")
        return vwap_path, None
    logger.info(f"更新数据时间范围:{update_start_date} to {update_end_date}")
    _files = [filename for filename in os.listdir(data_1min_path) if filename.endswith('.feather')]
    if "sp1000.feather" in _files:
        _files.remove("sp1000.feather")
    if len(_files) == 0:
        logger.error(f"{data_1min_path}不存在分钟级别数据")
        return vwap_path, None
    
    datas = Parallel(n_jobs=jobs)(delayed(_calc_vwap)(fn, data_1min_path, update_start_date) for fn in _files)
    df_new_vwap_daily_prices = pd.concat(datas)
    # update_end_trddt = df_new_vwap_daily_prices.index.get_level_values(1).max()
    df_all_vwap_daily_prices = pd.concat([df_local_stock_post_daily_vwap, df_new_vwap_daily_prices]).sort_index()
    df_all_vwap_daily_prices = df_all_vwap_daily_prices.reset_index()
    start_time = time.time()
    df_all_vwap_daily_prices.to_feather(vwap_path)
    end_time = time.time()
    logger.info(f"完成日度VWAP数据更新, {vwap_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_new_vwap_daily_prices)/(end_time-start_time), 2)}row/s")
    return vwap_path, df_new_vwap_daily_prices


def _calc_twap(file_name, data_1min_path, update_start_date):
    update_start_date = pd.to_datetime(update_start_date)
    _fp = os.path.join(data_1min_path, file_name)
    # 增加时间约束，只计算更新部分的日度twap
    _df = pd.read_feather(_fp)
    _df.set_index(['datetime'], inplace=True)
    try:
        _df = _df.loc[_df.index > update_start_date].reset_index()
    except:
        print(_df)
        raise KeyError(f'{update_start_date} {_fp}')
    """_df.columns
    Index(['close', 'low', 'total_turnover', 'num_trades', 'volume', 'open', 'high'], dtype='object')
    """
    # 分钟级数据中所有列都是数值，可以直接mean
    # 这里这样提取date，意味着通过min数据计算的daily vwap数据的label=='left'，即T日的数据代表的是[T, T+1)的数据，closed=='left'
    _df['date'] = _df['datetime'].dt.strftime('%Y%m%d')
    _df.drop(columns=['datetime'], inplace=True)
    _df = _df.groupby(by=['date']).mean(numeric_only=True).reset_index()

    _order_book_id = file_name.split('.feather')[0]
    _df['order_book_id'] = _order_book_id
    _df.set_index(['order_book_id', 'date'], inplace=True)
    return _df


def update_stock_daily_twap_prices(data_1min_path, tvap_path, jobs=8):
    """
    基于分钟级数据(增量)更新本地TWAP数据
        - TWAP: 时间加权价格=mean([P_1, P_2, ..., P_n])
    """
    if not os.path.exists(os.path.dirname(data_1min_path)):
        os.makedirs(os.path.dirname(data_1min_path))
    if not os.path.exists(os.path.dirname(tvap_path)):
        os.makedirs(os.path.dirname(tvap_path))
    logger.info("启动个股日度TWAP数据(后复权)更新")
    adjust_type = 'post'
    # 首先读取本地储存的后复权TWAP价格数据
    # 如果已存在文件，则增量更新
    if os.path.exists(tvap_path):
        df_local_stock_post_daily_twap = pd.read_feather(tvap_path)
        df_local_stock_post_daily_twap.set_index(['order_book_id', 'date'], inplace=True)
        """df_local_stock_post_daily_twap.head(3)
                                     close         low  ...        open        high
        order_book_id date                              ...                        
        000001.XSHE   20050104  215.822909  215.644851  ...  215.832571  216.023051
                      20050105  213.347993  213.163031  ...  213.354897  213.510870
                      20050106  215.263890  215.123093  ...  215.266650  215.393641
        """
        # 确定更新的起点：以默认个股000001的行情为准
        code = df_local_stock_post_daily_twap.index.to_list()[0][0]
        local_max_date = df_local_stock_post_daily_twap.loc[code].index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)
    # 如果无文件，则说明是首次生成，直接设置起始结束时间点即可
    else:
        df_local_stock_post_daily_twap = pd.DataFrame()
        update_start_date = pd.to_datetime('2000-01-04')
        logger.warning(f"此前尚未生成TWAP数据在路径:{tvap_path} 将从默认起点:{update_start_date}开始生成数据,耗时较久")

    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{tvap_path}当前已是最新数据，跳过 {update_start_date} to {update_end_date}")
        return tvap_path, None
    
    logger.info(f"计划更新数据时间范围:{update_start_date} to {update_end_date}")
    _files = [filename for filename in os.listdir(data_1min_path) if filename.endswith('.feather')]
    if "sp1000.feather" in _files:
        _files.remove("sp1000.feather")
    if len(_files) == 0:
        logger.error(f"{data_1min_path}不存在分钟级别数据")
        return tvap_path, None
    
    datas = Parallel(n_jobs=jobs)(delayed(_calc_twap)(fn, data_1min_path, update_start_date) for fn in _files)
    df_new_twap_daily_prices = pd.concat(datas)
    # df_new_twap_daily_prices.head(3)
    # update_end_trddt = df_new_twap_daily_prices.index.get_level_values(1).max()
    df_all_twap_daily_prices = pd.concat([df_local_stock_post_daily_twap, df_new_twap_daily_prices]).sort_index()
    df_all_twap_daily_prices = df_all_twap_daily_prices.reset_index()
    start_time = time.time()
    df_all_twap_daily_prices.to_feather(tvap_path)
    end_time = time.time()
    logger.info(f"完成个股日度TWAP数据更新, {tvap_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_new_twap_daily_prices)/(end_time-start_time), 2)}row/s")
    return tvap_path, df_new_twap_daily_prices

def _aggregate(file_name, data_1min_path, update_start_date, frequency, data_frequency_path):
    update_start_date = pd.to_datetime(update_start_date)
    _fp = os.path.join(data_1min_path, file_name)
    _df = pd.read_feather(_fp)
    _df.set_index(['datetime'], inplace=True)
    _df = _df.loc[_df.index > update_start_date]
    _df['vwap'] = _df['close'] * _df['volume']
    _df['twap'] = _df['close']
    label = 'right'
    if frequency in ['1d', ]:
        label = 'left'
    if 'num_trades' in _df.columns:
        _df_rsmpl = _df.resample(
            frequency, label=label, closed='right',
        ).agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last',
            'vwap': 'sum', 'twap': 'mean',
            'volume': 'sum', 'total_turnover': 'sum',
            'num_trades': 'sum'
        })
    else:
        # 指数没有num_trades，不进行num_trades加总
        _df_rsmpl = _df.resample(
            frequency, label=label, closed='right',
        ).agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last',
            'vwap': 'sum', 'twap': 'mean',
            'volume': 'sum', 'total_turnover': 'sum',
        })
    _df_rsmpl['vwap'] = _df_rsmpl['vwap']/_df_rsmpl['volume']
    # 如果OHLC同时为NaN，表明不是交易时间，直接删除
    _df_rsmpl.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
    # 导出
    _fp_out = os.path.join(data_frequency_path, file_name)
    
    if len(_df_rsmpl) > 0:
        if os.path.exists(_fp_out):
            df_local = pd.read_feather(_fp_out)
            df_local.set_index(['datetime'], inplace=True)
            df_all = pd.concat([df_local, _df_rsmpl]).sort_index()
        else:
            df_all = _df_rsmpl
        start_time = time.time()
        df_all.reset_index(inplace=True)
        df_all.to_feather(_fp_out)
        end_time = time.time()
        logger.info(f"完成{frequency}级别数据更新, {_fp_out}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all)/(end_time-start_time), 2)}row/s")
    else:
        logger.warning(f"{_fp_out}没有最新数据需要更新，{file_name.split('.feather')[0]}可能已经停牌或退市")
    return _fp_out, _df_rsmpl

def update_stock_vwap_twap_xmin_prices(data_1min_path, data_frequency_path, frequency, order_book_id='000001.XSHE', jobs=8):
    """
    不同频率下VWAP和TWAP数据更新
    """
    if not os.path.exists(data_1min_path):
        os.makedirs(data_1min_path)
    if not os.path.exists(data_frequency_path):
        os.makedirs(data_frequency_path)
    logger.info(f"启动个股{frequency}级别VWAP、TWAP数据(后复权)更新")
     
    # 更新本地已包含的个股数据
    def _read_local_max_date(file, data_1min_path, data_frequency_path):
        fp_1min_path = os.path.join(data_1min_path, file)
        fp_xmin_path = os.path.join(data_frequency_path, file)
        if os.path.exists(fp_1min_path) and os.path.exists(fp_xmin_path):
            # 读取最后100条记录
            try:
                _df = pd.read_feather(fp_xmin_path)
                _df.set_index(['datetime'], inplace=True)
                _local_max_date = _df.index.max()
            except:
                raise KeyError(f"{fp_xmin_path} open failed")
        else:
            _local_max_date = datetime(2000, 1, 4)
        return file, _local_max_date

    _files = [filename for filename in os.listdir(data_1min_path) if filename.endswith('.feather')]
    if "sp1000.feather" in _files:
        _files.remove("sp1000.feather")

    local_max_dates = Parallel(n_jobs=jobs)(delayed(_read_local_max_date)(file, data_1min_path, data_frequency_path) for file in _files)
    df_local_max_dates = pd.DataFrame(local_max_dates, columns=['file', 'local_max_date'])
    uniq_local_max_date = df_local_max_dates['local_max_date'].unique()
    if len(uniq_local_max_date) > 1:
        logger.warning(f"本地股票数据的最新日期不同，需要分{len(uniq_local_max_date)}批次进行数据更新.")
    results = list()
    for local_max_date in uniq_local_max_date:
        file_list = df_local_max_dates[df_local_max_dates['local_max_date'] == local_max_date]['file'].to_list()
        file_list = list(set(file_list))
        # 更新起点：本地最新日期的下一个交易日
        update_start_trddt = rqdatac.get_next_trading_date(local_max_date)  
        # 更新终点：最新交易日的前一个交易日
        now = datetime.now()
        if now.hour >= 16:
            update_end_trddt = rqdatac.get_latest_trading_date()
        else:
            update_end_trddt = rqdatac.get_previous_trading_date(now)
        if update_start_trddt > update_end_trddt:
            logger.info(f"{local_max_date.astype('datetime64[D]')}批次当前已是最新数据，跳过 {update_start_trddt} to {update_end_trddt}")
            continue
        logger.info(f"计划更新数据时间范围:{update_start_trddt} to {update_end_trddt}")
        if jobs > len(file_list):
            jobs = round(len(file_list) / 2) + 1
        result = Parallel(n_jobs=jobs)(delayed(_aggregate)(fn, data_1min_path, update_start_trddt, frequency, data_frequency_path) for fn in file_list)
        results.extend(result)
    return results

def update_stock_vwap_twap_prices(data_1min_path, data_frequency_path, frequency, order_book_id='000001.XSHE', jobs=8):
    """
    不同频率下VWAP和TWAP数据更新
    """
    if not os.path.exists(data_1min_path):
        os.makedirs(data_1min_path)
    if not os.path.exists(data_frequency_path):
        os.makedirs(data_frequency_path)
    logger.info(f"启动个股{frequency}级别VWAP、TWAP数据(后复权)更新")
    # adjust_type = 'post'  # 默认后复权数据
    if not os.path.exists(os.path.join(data_frequency_path, f'{order_book_id}.feather')):
        update_start_date = pd.to_datetime('2005-01-04')
    else:
        # 通过start限制范围，但要求数据以顺序排列
        # 内存和性能充足的情况下有也可以直接全部读取
        filepath = os.path.join(data_frequency_path, f'{order_book_id}.feather')
        df_default = pd.read_feather(filepath)
        df_default.set_index(['datetime'], inplace=True)
        """df_default.head(3)
                                close       low  ...      open      high
        datetime                                 ...                    
        2005-01-04 09:31:00  217.6477  217.6477  ...  218.3102  218.3102
        2005-01-04 09:32:00  217.3164  217.3164  ...  217.6477  217.6477
        2005-01-04 09:33:00  217.3164  216.9851  ...  217.3164  217.3164
        """
        # 确定更新的起点：以默认个股000001.XSHE平安银行的行情为准
        local_max_date = df_default.index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)

    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{data_frequency_path}当前已是最新数据，跳过 {update_start_date} to {update_end_date}")
        return []
    
    logger.info(f"计划更新数据时间范围:{update_start_date} to {update_end_date}")
    _files = [filename for filename in os.listdir(data_1min_path) if filename.endswith('.feather')]
    if "sp1000.feather" in _files:
        _files.remove("sp1000.feather")
    results = Parallel(n_jobs=jobs)(delayed(_aggregate)(fn, data_1min_path, update_start_date, frequency, data_frequency_path) for fn in _files)
    return results

def _get_daily_windows(frequency, days):
    one_day_min = 240
    freq_num = int(frequency.replace('min', ''))
    window_num = days * one_day_min / freq_num
    return int(window_num)


def extend_index1000_prices_by_index500(data_1min_path, data_frequency_path, frequency, preset_days=(2, 3, 5), close='close'):
    """
    使用中证500的分钟数据延长至2010年以前
        - 由于价格不可比,所以只能先算出ret保存下来
    """
    if not os.path.exists(os.path.dirname(data_1min_path)):
        os.makedirs(os.path.dirname(data_1min_path))
    logger.info(f"开始合成{frequency}级别sp1000行情数据")
    if frequency == '1min':
        data_frequency_path = data_1min_path

    fp_zz1000 = os.path.join(data_frequency_path, '000852.XSHG.feather')
    # 读取中证1000行情数据
    if not os.path.exists(fp_zz1000):
        logger.error(f"未找到本地中证1000行情数据:{fp_zz1000}")
        return None, None
    _zz1000_prices = pd.read_feather(fp_zz1000)
    _zz1000_prices.set_index(['datetime'], inplace=True)
    _zz1000_prices[f'{close}___roc__window_1'] = _zz1000_prices[close].pct_change()
    for days in preset_days:
        window = _get_daily_windows(frequency, days)
        _zz1000_prices[f'{close}___roc__window_{window}'] = _zz1000_prices[close].pct_change(window)

    # _zz1000_start_date = _zz1000_prices.index.min()
    _zz1000_start_date = '2015-01-01'
    _zz1000_prices_after = _zz1000_prices[_zz1000_prices.index >= _zz1000_start_date]

    fp_zz500 = os.path.join(data_1min_path, '000905.XSHG.feather')
    if not os.path.exists(fp_zz500):
        logger.error(f"未找到本地中证500行情数据:{fp_zz500}")
        return None, None
    # 读取中证500行情数据
    _zz500_prices = pd.read_feather(fp_zz500)
    _zz500_prices.set_index(['datetime'], inplace=True)
    _zz500_prices[f'{close}___roc__window_1'] = _zz500_prices['close'].pct_change()
    for days in preset_days:
        window = _get_daily_windows(frequency, days)
        _zz500_prices[f'{close}___roc__window_{window}'] = _zz500_prices[close].pct_change(window)

    _zz500_prices_prev = _zz500_prices[_zz500_prices.index < _zz1000_start_date]

    df_all = pd.concat([_zz500_prices_prev, _zz1000_prices_after]).sort_index()
    _filename = "sp1000.feather"
    fp_sp1000 = os.path.join(data_frequency_path, _filename)
    start_time = time.time()
    df_all.reset_index(inplace=True)
    df_all.to_feather(fp_sp1000)
    end_time = time.time()
    logger.info(f"合成{frequency}级别sp1000行情数据完成, {fp_sp1000}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all)/(end_time-start_time), 2)}row/s")
    return fp_sp1000, df_all
