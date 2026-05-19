import pandas as pd
import os
from datetime import datetime
import time
from loguru import logger # type: ignore
import rqdatac # type: ignore


def update_stocks_basic_info(basic_data_path):
    """
    下载个股基本信息
    """
    if not os.path.exists(os.path.dirname(basic_data_path)):
        os.makedirs(os.path.dirname(basic_data_path))
    logger.info("开始股票基本信息更新")
    df_basics = rqdatac.all_instruments(type='CS', market='cn', date=None)
    df_basics.set_index(['order_book_id'], inplace=True)
    df_basics.index.name = 'order_book_id' 
    df_basics.reset_index(inplace=True)
    start_time = time.time()
    df_basics.to_feather(basic_data_path)
    end_time = time.time()
    logger.info(f"股票基本信息更新完成, {basic_data_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_basics)/(end_time-start_time), 2)}row/s")
    return basic_data_path, df_basics


def download_st_stock_info(st_path):
    """
    本地初始化个股ST状态矩阵
    """
    if not os.path.exists(os.path.dirname(st_path)):
        os.makedirs(os.path.dirname(st_path))
    logger.info("开始ST股票基本信息更新")
    # 初始化起始日期为上证开市日
    start_date = '1990-12-19'
    end_date = pd.to_datetime(rqdatac.get_latest_trading_date())
    _df_all_order_book_ids_info = rqdatac.all_instruments(type='CS', market='cn', date=None)
    _order_book_ids = _df_all_order_book_ids_info['order_book_id'].unique().tolist()
    _df_st_all = rqdatac.is_st_stock(_order_book_ids, start_date, end_date)
    _df_st_all.index.name = 'date' 
    _df_st_all.reset_index(inplace=True)
    start_time = time.time()
    _df_st_all.to_feather(st_path)
    end_time = time.time()
    logger.info(f"ST股票信息更新完成, {st_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(_df_st_all)/(end_time-start_time), 2)}row/s")
    return st_path, _df_st_all


def download_suspended_stock_info(suspend_path):
    """
    本地初始化个股停牌状态矩阵
    """
    if not os.path.exists(os.path.dirname(suspend_path)):
        os.makedirs(os.path.dirname(suspend_path))
    logger.info("开始停牌股票基本信息更新")
    # 初始化起始日期为上证开市日，
    start_date = '1990-12-19'
    end_date = pd.to_datetime(rqdatac.get_latest_trading_date())
    _df_all_order_book_ids_info = rqdatac.all_instruments(type='CS', market='cn', date=None)
    _order_book_ids = _df_all_order_book_ids_info['order_book_id'].unique().tolist()
    _df_susp_all = rqdatac.is_suspended(_order_book_ids, start_date, end_date)
    _df_susp_all.index.name = 'date' 
    _df_susp_all.reset_index(inplace=True)
    start_time = time.time()
    _df_susp_all.to_feather(suspend_path)
    end_time = time.time()
    logger.info(f"停牌股票信息更新完成, {suspend_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(_df_susp_all)/(end_time-start_time), 2)}row/s")
    return suspend_path, _df_susp_all


def update_index_components(data_path, to_update_order_book_ids=None):
    """
    更新指数成分股数据
        目前默认更新沪深300,中证500,中证800,中证1000和中证全指的数据
        指数成分股的更新是增量式更新,所以对更新时间有约束,一般是当日15:00-23:59之间更新。
    """
    result = dict()
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logger.info("开始指数成分股数据更新")
    if to_update_order_book_ids is None:
        to_update_order_book_ids = ['000016.XSHG', '000300.XSHG', '000905.XSHG', '000906.XSHG', '000852.XSHG', '000985.XSHG', '932000.INDX', '000510.XSHG']
        logger.info(f"默认更新以下指数成分股: {to_update_order_book_ids}")
    DICT_RQ_INDEX_COMPONETS_START_DATE = {
        '000016.XSHG': '2004-01-02',  # 上证50
        '000300.XSHG': '2005-04-08',  # 沪深300
        '000905.XSHG': '2007-01-15',  # 中证500
        '000906.XSHG': '2007-01-15',  # 中证800
        '000852.XSHG': '2014-10-17',  # 中证1000
        '932000.INDX': '2023-08-11',  # 中证2000
        '000985.XSHG': '2011-08-02',  # 中证全指
        '000510.XSHG': '2024-09-23',  # 中证A500
    }
    df_dict = dict()
    for order_book_id in to_update_order_book_ids:
        update_start_date = pd.to_datetime(DICT_RQ_INDEX_COMPONETS_START_DATE[order_book_id])
        now = datetime.now()
        if now.hour >= 16:
            update_end_date = rqdatac.get_latest_trading_date()
        else:
            update_end_date = rqdatac.get_previous_trading_date(now)
        
        logger.info(f"更新{order_book_id}指数 时间范围：{update_start_date} to {update_end_date}")
        results = rqdatac.index_components(order_book_id, start_date=update_start_date, end_date=update_end_date)
        # 存在个数不同的情况（例如沪深300首日只有299只股票）
        dfs = []
        for dt, order_book_id_list in results.items():
            df = pd.DataFrame({'date': dt, 'order_book_id': order_book_id_list, 'state': 1})
            dfs.append(df)

        df = pd.concat(dfs).sort_values(by=['date', 'order_book_id'])
        df_all_index_comp = df.reset_index(drop=True).pivot(index='date', columns='order_book_id', values='state') 
        df_all_index_comp.columns.name = None
        df_dict[order_book_id] = df_all_index_comp
        df_all_index_comp.reset_index(inplace=True)
        filepath = os.path.join(data_path, f"index_components_{order_book_id}.feather")
        start_time = time.time()
        df_all_index_comp.to_feather(filepath)
        end_time = time.time()
        logger.info(f"{order_book_id}指数成分股数据更新完成, {filepath}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_index_comp)/(end_time-start_time), 2)}row/s")

        result[order_book_id] = (data_path, df_all_index_comp)

    # 添加中证1800（沪深300+中证500+中证1000）方便使用
    st_dt = '2015-01-01'
    df_zz800 = df_dict['000906.XSHG'].loc[st_dt:]
    df_zz1000 = df_dict['000852.XSHG'].loc[st_dt:]
    df_zz1800 = df_zz800.combine_first(df_zz1000)
    df_dict['zz1800'] = df_zz1800
    result['zz1800'] = (data_path, df_zz1800)
    df_zz1800.reset_index(inplace=True)
    filepath = os.path.join(data_path, f"index_components_zz1800.feather")
    start_time = time.time()
    df_zz1800.to_feather(filepath)
    end_time = time.time()
    logger.info(f"zz1800指数成分股数据更新完成, {filepath}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_zz1800)/(end_time-start_time), 2)}row/s")

    # 添加中证1500（中证1000+中证500）
    st_dt = '2015-01-01'
    df_zz500 = df_dict['000905.XSHG'].loc[st_dt:]
    df_zz1000 =  df_dict['000852.XSHG'].loc[st_dt:]
    df_zz1500 = df_zz500.combine_first(df_zz1000)
    df_dict['zz1500'] = df_zz1500
    result['zz1500'] = (data_path, df_zz1500)
    df_zz1500.reset_index(inplace=True)
    filepath = os.path.join(data_path, f"index_components_zz1500.feather")
    start_time = time.time()
    df_zz1500.to_feather(filepath)
    end_time = time.time()
    logger.info(f"zz1500指数成分股数据更新完成, {filepath}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_zz1500)/(end_time-start_time), 2)}row/s")

    logger.info("指数成分股数据更新完成")
    return result


def update_index_components_weights(data_path, order_book_ids=None):
    """
    更新指数成分股权重数据
    """
    result = dict()
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logger.info("启动指数成分股权重数据更新")
    if order_book_ids is None:
        DEFAULT_INDEX_ORDER_BOOK_IDS = ['000016.XSHG', '000300.XSHG', '000905.XSHG', '000906.XSHG', '932000.INDX', '000852.XSHG', '000510.XSHG']
        order_book_ids = DEFAULT_INDEX_ORDER_BOOK_IDS
        logger.info(f"默认更新以下指数成分股 {order_book_ids}")
        
    DICT_RQ_INDEX_COMPONETS_START_DATE = {
        '000016.XSHG': '2004-01-02',  # 上证50
        '000300.XSHG': '2005-04-08',  # 沪深300
        '000905.XSHG': '2007-01-15',  # 中证500
        '000906.XSHG': '2007-01-15',  # 中证800
        '000852.XSHG': '2014-10-17',  # 中证1000
        '932000.INDX': '2023-08-11',  # 中证2000
        '000985.XSHG': '2011-08-02',  # 中证全指
        '000510.XSHG': '2024-09-23',  # 中证A500
    }
    for order_book_id in order_book_ids:
        update_start_date = pd.to_datetime(DICT_RQ_INDEX_COMPONETS_START_DATE[order_book_id])
        now = datetime.now()
        if now.hour >= 16:
            update_end_date = rqdatac.get_latest_trading_date()
        else:
            update_end_date = rqdatac.get_previous_trading_date(now)
        if update_start_date > update_end_date:
            logger.info(f"{order_book_id}当前已是最新数据，跳过 {update_start_date} to {update_end_date}")
            continue
        
        logger.info(f"更新{order_book_id}指数权重 时间范围：{update_start_date} to {update_end_date}")
        df = rqdatac.index_weights_ex(order_book_id, start_date=update_start_date, end_date=update_end_date, market='cn')
        if df is None:
            logger.error(f"更新{order_book_id}指数失败 时间范围：{update_start_date} to {update_end_date}")
            continue
        df_all_index_comp_weights = df.reset_index().pivot(index='date', columns='order_book_id', values='weight')
        df_all_index_comp_weights.columns.name = None
        df_all_index_comp_weights.reset_index(inplace=True)
        file_path = os.path.join(data_path, f"index_components_weights_{order_book_id}.feather")
        start_time = time.time()
        df_all_index_comp_weights.to_feather(file_path)
        end_time = time.time()
        logger.info(f"{order_book_id}指数成分股权重数据更新完成, {file_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_index_comp_weights)/(end_time-start_time), 2)}row/s")
        result[order_book_id] = (data_path, df_all_index_comp_weights)
        
    logger.info("指数成分股权重数据更新完成")
    return result


def update_index_components_weights_industry(data_path, order_book_ids=None):
    """
    更新指数成分股权重数据、行业
    """
    result = dict()
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logger.info("启动指数成分股权重行业数据更新")
    if order_book_ids is None:
        DEFAULT_INDEX_ORDER_BOOK_IDS = ['000016.XSHG', '000300.XSHG', '000905.XSHG', '000906.XSHG', '932000.INDX', '000852.XSHG', '000510.XSHG']
        order_book_ids = DEFAULT_INDEX_ORDER_BOOK_IDS
        logger.info(f"默认更新以下指数成分股 {order_book_ids}")
        
    DICT_RQ_INDEX_COMPONETS_START_DATE = {
        '000016.XSHG': '2004-01-02',  # 上证50
        '000300.XSHG': '2005-04-08',  # 沪深300
        '000905.XSHG': '2007-01-15',  # 中证500
        '000906.XSHG': '2007-01-15',  # 中证800
        '000852.XSHG': '2014-10-17',  # 中证1000
        '932000.INDX': '2023-08-11',  # 中证2000
        '000985.XSHG': '2011-08-02',  # 中证全指
        '000510.XSHG': '2024-09-23',  # 中证A500
    }
    for order_book_id in order_book_ids:
        update_start_date = pd.to_datetime(DICT_RQ_INDEX_COMPONETS_START_DATE[order_book_id])
        now = datetime.now()
        if now.hour >= 16:
            update_end_date = rqdatac.get_latest_trading_date()
        else:
            update_end_date = rqdatac.get_previous_trading_date(now)
        if update_start_date > update_end_date:
            logger.info(f"{order_book_id}当前已是最新数据，跳过 {update_start_date} to {update_end_date}")
            continue
        
        logger.info(f"更新{order_book_id}指数权重 时间范围：{update_start_date} to {update_end_date}")
        df = rqdatac.index_weights_ex(order_book_id, start_date=update_start_date, end_date=update_end_date, market='cn')
        if df is None:
            logger.error(f"更新{order_book_id}指数失败 时间范围：{update_start_date} to {update_end_date}")
            continue
        df = df.reset_index().pivot(index='date', columns='order_book_id', values='weight').shift(1)
        df = df.stack(level=-1).reset_index()
        df.columns = ['date', 'order_book_id', 'weight']
        df = df.set_index(['date', 'order_book_id'])
        df_all_index_comp_weights = pd.DataFrame(index=['date', 'order_book_id'], columns=['date', 'order_book_id', 'weight', 'industry'])
        df_all_index_comp_weights = df_all_index_comp_weights.set_index(['date', 'order_book_id']).dropna()
        date_list = df.index.get_level_values('date').unique().tolist()
        for _date in date_list:
            order_book_ids = df.loc[_date].index.get_level_values('order_book_id').unique().tolist()
            industries = rqdatac.get_instrument_industry(order_book_ids=order_book_ids, date=_date)['first_industry_name']
            data = pd.concat([df.loc[_date], industries], axis=1, join="inner")
            data['industry'] = data['first_industry_name']
            del data['first_industry_name']
            data['date'] = _date
            data = data.reset_index()
            data = data.set_index(["date", "order_book_id"])
            df_all_index_comp_weights = pd.concat([df_all_index_comp_weights, data])
        df_all_index_comp_weights.reset_index(inplace=True)
        file_path = os.path.join(data_path, f"index_components_weights_industry_{order_book_id}.feather")
        start_time = time.time()
        df_all_index_comp_weights.to_feather(file_path)
        end_time = time.time()
        logger.info(f"{order_book_id}指数成分股权重行业数据更新完成, {file_path}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_index_comp_weights)/(end_time-start_time), 2)}row/s")
        result[order_book_id] = (data_path, df_all_index_comp_weights)
        
    logger.info("指数成分股权重行业数据更新完成")
    return result


def update_all_instruments_industry(industry_data_path:str):
    """
    更新股票行业数据
    """
    if os.path.exists(industry_data_path):
        # 已存在文件，增量更新
        df_all_instruments_industry = pd.read_feather(industry_data_path)
        df_all_instruments_industry = df_all_instruments_industry.set_index(['date', 'order_book_id']).dropna()
        if df_all_instruments_industry is None:
            raise ValueError(f"{industry_data_path} read failed")
        # 确定更新的起点：直接读取索引的最大值即可
        local_max_date = df_all_instruments_industry.index.get_level_values('date').max()
        start_date = rqdatac.get_next_trading_date(local_max_date)
    else:
        # 无文件，首次生成，直接设置默认起点
        df_all_instruments_industry = pd.DataFrame(index=['date', 'order_book_id'], columns=['date', 'order_book_id', 'industry'])
        df_all_instruments_industry = df_all_instruments_industry.set_index(['date', 'order_book_id']).dropna()
        start_date = pd.to_datetime('2003-01-01')

    now = datetime.now()
    if now.hour >= 16:
        end_date = rqdatac.get_latest_trading_date()
    else:
        end_date = rqdatac.get_previous_trading_date(now)

    if start_date > end_date:
        logger.info(f"股票行业数据已是最新数据，跳过 时间范围:{start_date} to {end_date}")
        return

    logger.info(f"启动股票行业数据更新 更新时间范围:{start_date} to {end_date}")

    date_list = rqdatac.get_trading_dates(start_date=start_date, end_date=end_date, market='cn')
    for _date in date_list:
        _date = _date.strftime('%Y-%m-%d')
        print(_date)
        df_basics = rqdatac.all_instruments(type='CS', market='cn', date=_date)
        order_book_ids = df_basics['order_book_id'].unique().tolist()
        industries = rqdatac.get_instrument_industry(order_book_ids=order_book_ids, source='citics_2019', level=1, date=_date, market='cn')
        industries['industry'] = industries['first_industry_name']
        industries['industry_code'] = industries['first_industry_code']
        del industries['first_industry_code']
        del industries['first_industry_name']

        industries['date'] = _date
        industries = industries.reset_index()
        industries = industries.set_index(["date", "order_book_id"])
        df_all_instruments_industry = pd.concat([df_all_instruments_industry, industries])
        
    print(df_all_instruments_industry)
    df_all_instruments_industry.reset_index(inplace=True)
    df_all_instruments_industry.to_feather(industry_data_path)
    logger.info(f"更新股票行业数据更新完成, {start_date} to {end_date} {industry_data_path}")
    return df_all_instruments_industry
