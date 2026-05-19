import os
import pandas as pd
# import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
import warnings
import os
import sys
from datetime import datetime
import time
# import jtfactor.factor as jtfactor
import ultis.jtfactor.factor as jtfactor
import rqdatac # type: ignore
from loguru import logger # type: ignore
warnings.filterwarnings('ignore')


def _calc_factor(file_name, source_dir_min_prices, update_data_start_date, window, base_step, by_date, _func, kwargs):
    update_data_start_date = pd.to_datetime(update_data_start_date)
    _order_book_id = file_name.split('.feather')[0]
    _fp = os.path.join(source_dir_min_prices, file_name)
    _df = pd.read_feather(_fp)
    _df.set_index(['datetime'], inplace=True)
    _df = _df.loc[_df.index > update_data_start_date]  # 增加时间约束，只计算更新部分的日度twap
    """_df.columns
    Index(['datetime', 'close', 'low', 'total_turnover', 'num_trades', 'volume', 'open', 'high'], dtype='object')
    _df.head(3)
                    datetime     close       low  ...    volume      open      high
    0 2005-01-04 09:31:00  217.6477  217.6477  ...  843.7008  218.3102  218.3102
    1 2005-01-04 09:32:00  217.3164  217.3164  ...  762.6674  217.6477  217.6477
    2 2005-01-04 09:33:00  217.3164  216.9851  ...  490.9672  217.3164  217.3164
    """
    # 空数据表明最近已停牌、退市等，跳过计算
    if _df.empty:
        return pd.Series(name=_order_book_id, dtype=float)
    if _df.shape[0] < window * 240 / base_step:
        return pd.Series(name=_order_book_id, dtype=float)
    # 注解2：此时的index时刻实际对应的是当时的结束（基于分钟级数据计算，自然继承分钟级的逻辑）
    try:
        if by_date:
            # 这里这样提取date，意味着通过min数据计算的daily vwap数据的label=='left'，即T日的数据代表的是[T, T+1)的数据，closed=='left'
            _factor = _df.groupby(_df.index.date, group_keys=False).apply(_func, **kwargs)
        else:
            # 若所调用的因子计算函数中本身就有关于date的处理（例如resample等），则这里边不需要by_date
            _factor = _func(_df, **kwargs)
    except:
        raise ValueError("break")
    _factor.name = _order_book_id
    return _factor

def _update_stock_daily_factor(data_1min_path, factor_post_1d_datapath, factor_name, jobs=8, debug=False, **kwargs):
    """
    基于分钟级数据(增量)更新本地日度因子数据
    - 增量更新，同样需要添加校验代码
    """
    # 20240321: 部分因子计算需要更长的历史数据（因子计算函数非日度、window超过1）
    _func = getattr(jtfactor, f"jt_{factor_name.upper()}")
    by_date = getattr(_func, 'by_date', True)
    window = getattr(_func, 'window', 0)
    base_frequency = getattr(_func, 'base_frequency', '1min')
    if not os.path.exists(factor_post_1d_datapath):
        os.makedirs(factor_post_1d_datapath)
    adjust_type = 'post'
    # 确认本地文件路径
    fp_factor = os.path.join(factor_post_1d_datapath, f"{factor_name}.feather")
    if os.path.exists(fp_factor):
        # 已存在文件，增量更新
        df_local_factor = pd.read_feather(fp_factor)
        df_local_factor.set_index(['date'], inplace=True)
        """df_local_factor.tail(3)
                    000001.XSHE  000002.XSHE  ...  688981.XSHG  689009.XSHG
        date                                  ...                          
        2024-03-15     0.001293     0.001212  ...     0.000920     0.002341
        2024-03-18     0.000828     0.001023  ...     0.001299     0.002136
        2024-03-19     0.000752     0.000911  ...     0.000853     0.002596
        """
        if df_local_factor is None:
            raise ValueError(f"{fp_factor} read failed")
        # 确定更新的起点：直接读取索引的最大值即可
        local_max_date = df_local_factor.index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)
    else:
        # 无文件，首次生成，直接设置默认起点
        df_local_factor = pd.DataFrame()
        update_start_date = pd.to_datetime('2005-01-04')
        logger.warning(f"此前尚未生成{factor_name}数据在路径:{fp_factor} 将从默认起点：{update_start_date}开始生成数据，耗时较久")

    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{factor_name}已是最新数据，跳过 时间范围:{update_start_date} to {update_end_date}")
        return fp_factor, factor_name, None
    
    logger.info(f"启动{factor_name}因子日度级别数据更新 基本数据频率：{base_frequency} 更新时间范围:{update_start_date} to {update_end_date} jobs:{jobs}")

    # 根据因子的计算逻辑确定使用数据起点（例如SDRVOL要求至少20日数据）
    update_data_start_date = update_start_date
    if (by_date is False) and (window >= 1):
        logger.info(f"因子{factor_name}需要{window}日前推数据")
        update_data_start_date = rqdatac.get_previous_trading_date(update_start_date, window)

    # 读取用于因子计算的基础数据的频率，默认一分钟1min
    base_step = int(base_frequency.split('min')[0])
    if base_frequency in [None, '1min']:
        source_dir_min_prices = data_1min_path
    else:
        source_dir_min_prices = data_1min_path.replace('1min', base_frequency)
    if not os.path.exists(source_dir_min_prices):
        logger.error("暂时不支持以下频率的K线读取: {}".format(source_dir_min_prices))
        raise ValueError("暂时不支持以下频率的K线读取: {}".format(source_dir_min_prices))
    _files = [filename for filename in os.listdir(source_dir_min_prices) if filename.endswith('.feather')]
    if "sp1000.feather" in _files:
        _files.remove("sp1000.feather")
    datas = Parallel(n_jobs=jobs)(delayed(_calc_factor)(fn, source_dir_min_prices, update_data_start_date, window, base_step, by_date, _func, kwargs) for fn in _files)
    
    df_new_factor = pd.concat(datas, axis=1).sort_index()
    # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
    df_new_factor.index = pd.to_datetime(df_new_factor.index)  
    date_end = kwargs.get('date_end', True)
    if date_end and len(df_new_factor) > 0 and (df_new_factor.index[-1].time() != "00:00"):
        # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
        df_new_factor.index = pd.to_datetime(df_new_factor.index.date)  

    # 仅保留新数据部分
    df_new_factor = df_new_factor.loc[update_start_date:]
    update_end_trddt = df_new_factor.index.max()

    df_all_factor_data = pd.concat([df_local_factor, df_new_factor]).sort_index()
    df_all_factor_data.index.name = 'date'
    df_all_factor_data.reset_index(inplace=True)
    start_time = time.time()
    df_all_factor_data.to_feather(fp_factor)
    end_time = time.time()
    logger.info(f"{factor_name}因子更新完成 {fp_factor}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_factor_data)/(end_time-start_time), 2)}row/s")
    return fp_factor, factor_name, df_new_factor


def update_daily_simple_factors(data_1min_path, factor_post_1d_datapath, factor_name, windows=(48, 240), jobs=8, debug=False):
    
    """
    因子更新函数模板
    """
    results = []
    if windows is not None:
        for window in windows:
            result = _update_stock_daily_factor(data_1min_path=data_1min_path, factor_post_1d_datapath=factor_post_1d_datapath, 
                                                  factor_name=factor_name.format(window=window), jobs=jobs, debug=debug)
            results.append(result)
    else:
        result = _update_stock_daily_factor(data_1min_path=data_1min_path, factor_post_1d_datapath=factor_post_1d_datapath, 
                                              factor_name=factor_name, jobs=jobs, debug=debug)
        results.append(result)
    return results


def _calc_compound_factor(file_name, source_dir_min_prices, update_data_start_date, window, base_step, df_benchmark, benchmark, by_date, _func, kwargs):
    update_data_start_date = pd.to_datetime(update_data_start_date)
    _order_book_id = file_name.split('.feather')[0]
    _fp = os.path.join(source_dir_min_prices, file_name)
    try:
        _df = pd.read_feather(_fp)
        _df.set_index(['datetime'], inplace=True)
        _df['pct_change'] = _df['close'].pct_change()
        _df = _df.loc[_df.index > update_data_start_date]  # 增加时间约束，只计算更新部分的日度twap
        # 空数据表明已停牌等，跳过计算
        if _df.empty:
            return pd.Series(name=_order_book_id, dtype=float)
        if _df.shape[0] < window * 240 / base_step:
            return pd.Series(name=_order_book_id, dtype=float)
        # _df.drop_duplicates(inplace=True)
        # df_benchmark.drop_duplicates(inplace=True)
        _df['bm_pct_change'] = df_benchmark['pct_change']
        if benchmark == 'sp1000':
            _df = _df.merge(df_benchmark.filter(regex='___roc__'), left_index=True, right_index=True, how='left')
        
        # print(_df.head(3))
        # 注解2：此时的index时刻实际对应的是当时的结束（基于分钟级数据计算，自然继承分钟级的逻辑） 
    
        if by_date:
            # 这里这样提取date，意味着通过min数据计算的daily vwap数据的label=='left'，即T日的数据代表的是[T, T+1)的数据，closed=='left'
            _factor = _df.groupby(_df.index.date, group_keys=False).apply(_func, **kwargs)
        else:
            # 若所调用的因子计算函数中本身就有关于date的处理（例如resample等），则这里边不需要by_date
            _factor = _func(_df, **kwargs)
    except:
        print(file_name, _df.shape, benchmark, df_benchmark.shape)
        print(_df)
        print(df_benchmark)
        raise KeyError(_fp)
    # print(_factor.head(3))
    _factor.name = _order_book_id
    return _factor

def _update_stock_daily_compound_factor(data_1min_path, factor_post_1d_datapath, factor_name, benchmark='000852.XSHG', jobs=8, debug=False, **kwargs):
    
    adjust_type = 'post'
    # 20240321: 部分因子计算需要更长的历史数据（因子计算函数非日度、window超过1）
    _func = getattr(jtfactor, f"jt_{factor_name.upper()}")
    by_date = getattr(_func, 'by_date', True)
    window = getattr(_func, 'window', 0)
    frequency = getattr(_func, 'base_frequency', '1min')
    if not os.path.exists(factor_post_1d_datapath):
        os.makedirs(factor_post_1d_datapath)
    logger.info(f"启动{factor_name}复合因子日度级别数据更新 基本数据频率：{frequency}  benchmark:{benchmark} 并发数:{jobs}")
    # 确认本地文件路径
    fp_factor = os.path.join(factor_post_1d_datapath, f"{factor_name}_{benchmark}.feather")
    if os.path.exists(fp_factor):
        # 已存在文件，增量更新
        # df_local_factor = pd.read_feather(fp_factor, key=factor_name)
        df_local_factor = pd.read_feather(fp_factor)
        df_local_factor.set_index(['date'], inplace=True)
        # 确定更新的起点：本地储存格式为宽面板的parquet文件，直接读取索引的最大值即可
        local_max_date = df_local_factor.index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)
    else:
        # 无文件，首次生成，直接设置默认起点
        df_local_factor = pd.DataFrame()
        if benchmark == '000852.XSHG':
            update_start_date = pd.to_datetime('2014-10-17')
        elif benchmark == '000905.XSHG':
            update_start_date = pd.to_datetime('2007-01-15')
        elif benchmark == 'sp1000':
            update_start_date = pd.to_datetime('2007-01-15')
        else:
            update_start_date = pd.to_datetime('2005-01-04')
        logger.warning(f"{factor_name}因子尚未生成过，{fp_factor} benchmark:{benchmark} 将从默认起点:{update_start_date}开始生成数据，耗时较久")
        
    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{factor_name}已是最新数据, benchmark:{benchmark}, 跳过 时间范围:{update_start_date} to {update_end_date}")
        return fp_factor, factor_name, None

    logger.info(f"更新时间范围:{update_start_date} to {update_end_date}")
    update_data_start_date = update_start_date
    if (by_date is False) and (window >= 1):
        logger.info(f"因子{factor_name}需要{window}日前推数据 benchmark:{benchmark}")
        update_data_start_date = rqdatac.get_previous_trading_date(update_start_date, window)

    base_step = int(frequency.split('min')[0])
    if frequency in [None, '1min']:
        source_dir_min_prices = data_1min_path
    else:
        source_dir_min_prices = data_1min_path.replace('1min', frequency)

    if not os.path.exists(source_dir_min_prices):
        logger.error(f"暂时不支持以下频率的K线读取:{source_dir_min_prices} {factor_name} benchmark:{benchmark}")
        raise ValueError("暂时不支持以下频率的K线读取:{}".format(source_dir_min_prices))
    _files = [filename for filename in os.listdir(source_dir_min_prices) if filename.endswith('.feather')]
    # 涉及指数的指标，指数本身不再计算相关指标（这里可能没有完全删除，后续可以通过是否包含num_trades这一列作为判断依据）
    _files = list(set(_files) - {"000300.XSHG.feather", "000905.XSHG.feather", "000906.XSHG.feather", "000852.XSHG.feather", "000985.XSHG.feather",
                                 "sp1000.feather"})

    df_benchmark = pd.read_feather(os.path.join(source_dir_min_prices, f'{benchmark}.feather'))
    df_benchmark.set_index(['datetime'], inplace=True)
    update_data_start_date = pd.to_datetime(update_data_start_date)
    df_benchmark = df_benchmark.loc[df_benchmark.index > update_data_start_date]
    if benchmark != 'sp1000':
        df_benchmark['pct_change'] = df_benchmark['close'].pct_change()
    else:
        df_benchmark['pct_change'] = df_benchmark['close___roc__window_1']

    datas = Parallel(n_jobs=jobs)(delayed(_calc_compound_factor)(fn, source_dir_min_prices, update_data_start_date, window, base_step, df_benchmark, benchmark, by_date, _func, kwargs) for fn in _files)

    df_new_factor = pd.concat(datas, axis=1).sort_index()
    # 将分钟end转为日期end
    df_new_factor.index = df_new_factor.index.astype('datetime64[ns]')  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
    if kwargs.get('date_end', False):
        df_new_factor.index = pd.to_datetime(df_new_factor.index)  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
    else:
        df_new_factor.index = pd.to_datetime(df_new_factor.index.date)  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）

    # 仅保留新数据部分
    df_new_factor = df_new_factor.loc[update_start_date:]
    update_end_trddt = df_new_factor.index.max()
    df_all_factor_data = pd.concat([df_local_factor, df_new_factor]).sort_index()
    df_all_factor_data.index.name = 'date'
    # print(df_all_factor_data)
    df_all_factor_data.reset_index(inplace=True)
    start_time = time.time()
    df_all_factor_data.to_feather(fp_factor)
    end_time = time.time()
    logger.info(f"{factor_name}因子更新完成 {fp_factor}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_factor_data)/(end_time-start_time), 2)}row/s")
    return fp_factor, factor_name, df_new_factor


def update_daily_compound_factors(data_1min_path, factor_post_1d_datapath, factor_name, windows=(240, 480), benchmark="000852.XSHG", jobs=8, debug=False, **kwargs):
    """
    因子更新函数模板
    """
    results = []
    if windows is not None:
        for window in windows:
            result = _update_stock_daily_compound_factor(data_1min_path=data_1min_path, factor_post_1d_datapath=factor_post_1d_datapath, 
                                                        factor_name=factor_name.format(window=window), jobs=jobs, debug=debug, benchmark=benchmark, **kwargs)
            results.append(result)
    else:
        result = _update_stock_daily_compound_factor(data_1min_path=data_1min_path, fatacor_xmifactor_post_1d_datapathn_path=factor_post_1d_datapath, 
                                                    factor_name=factor_name, jobs=jobs, debug=debug, benchmark=benchmark, **kwargs)
        results.append(result)
    return results

def _update_compound_factor_data_xmin(data_1min_path, fatacor_xmin_path, factor_name, frequency, benchmark='000852.XSHG', jobs=8, debug=False, **kwargs):
    
    adjust_type = 'post'
    # 20240321: 部分因子计算需要更长的历史数据（因子计算函数非日度、window超过1）
    _func = getattr(jtfactor, f"jt_{factor_name.upper()}")
    by_date = getattr(_func, 'by_date', True)
    window = getattr(_func, 'window', 0)
    if not os.path.exists(fatacor_xmin_path):
        os.makedirs(fatacor_xmin_path)
    logger.info(f"启动个股{factor_name}复合因子{frequency}级别数据更新 benchmark:{benchmark} 并发数:{jobs}")
    # 确认本地文件路径
    fp_factor = os.path.join(fatacor_xmin_path, f"{factor_name}_{benchmark}.feather")
    if os.path.exists(fp_factor):
        # 已存在文件，增量更新
        # df_local_factor = pd.read_feather(fp_factor, key=factor_name)
        df_local_factor = pd.read_feather(fp_factor)
        df_local_factor.set_index(['datetime'], inplace=True)
        # 确定更新的起点：本地储存格式为宽面板的parquet文件，直接读取索引的最大值即可
        local_max_date = df_local_factor.index.max()
        update_start_date = rqdatac.get_next_trading_date(local_max_date)
    else:
        # 无文件，首次生成，直接设置默认起点
        df_local_factor = pd.DataFrame()
        if benchmark == '000852.XSHG':
            update_start_date = pd.to_datetime('2014-10-17')
        elif benchmark == '000905.XSHG':
            update_start_date = pd.to_datetime('2007-01-15')
        elif benchmark == 'sp1000':
            update_start_date = pd.to_datetime('2007-01-15')
        else:
            update_start_date = pd.to_datetime('2005-01-04')
        logger.warning(f"{factor_name}因子尚未生成过，{fp_factor} benchmark:{benchmark} 将从默认起点:{update_start_date}开始生成数据，耗时较久")
        
    now = datetime.now()
    if now.hour >= 16:
        update_end_date = rqdatac.get_latest_trading_date()
    else:
        update_end_date = rqdatac.get_previous_trading_date(now)
    if update_start_date > update_end_date:
        logger.info(f"{factor_name}已是最新数据, benchmark:{benchmark}, 跳过 时间范围:{update_start_date} to {update_end_date}")
        return fp_factor, factor_name, None

    logger.info(f"更新时间范围:{update_start_date} to {update_end_date}")
    update_data_start_date = update_start_date
    if (by_date is False) and (window >= 1):
        logger.info(f"因子{factor_name}需要{window}日前推数据 benchmark:{benchmark}")
        update_data_start_date = rqdatac.get_previous_trading_date(update_start_date, window)

    base_step = int(frequency.split('min')[0])
    if frequency in [None, '1min']:
        source_dir_min_prices = data_1min_path
    else:
        source_dir_min_prices = data_1min_path.replace('1min', frequency)

    if not os.path.exists(source_dir_min_prices):
        logger.error(f"暂时不支持以下频率的K线读取:{source_dir_min_prices} {factor_name} benchmark:{benchmark}")
        raise ValueError("暂时不支持以下频率的K线读取:{}".format(source_dir_min_prices))
    _files = [filename for filename in os.listdir(source_dir_min_prices) if filename.endswith('.feather')]
    # 涉及指数的指标，指数本身不再计算相关指标（这里可能没有完全删除，后续可以通过是否包含num_trades这一列作为判断依据）
    _files = list(set(_files) - {"000300.XSHG.feather", "000905.XSHG.feather", "000906.XSHG.feather", "000852.XSHG.feather", "000985.XSHG.feather",
                                 "sp1000.feather"})

    df_benchmark = pd.read_feather(os.path.join(source_dir_min_prices, f'{benchmark}.feather'))
    df_benchmark.set_index(['datetime'], inplace=True)
    update_data_start_date = pd.to_datetime(update_data_start_date)
    df_benchmark = df_benchmark.loc[df_benchmark.index > update_data_start_date]
    if benchmark != 'sp1000':
        df_benchmark['pct_change'] = df_benchmark['close'].pct_change()
    else:
        df_benchmark['pct_change'] = df_benchmark['close___roc__window_1']

    datas = Parallel(n_jobs=jobs)(delayed(_calc_compound_factor)(fn, source_dir_min_prices, update_data_start_date, window, base_step, df_benchmark, benchmark, by_date, _func, kwargs) for fn in _files)

    df_new_factor = pd.concat(datas, axis=1).sort_index()
    # 将分钟end转为日期end
    df_new_factor.index = df_new_factor.index.astype('datetime64[ns]')  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
    if benchmark == 'sp1000':
        df_new_factor.index = pd.to_datetime(df_new_factor.index)  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）
    else:
        df_new_factor.index = pd.to_datetime(df_new_factor.index.date)  # pd.to_datetime默认是当时的开始，例如 2024-03-19 00:00:00，与实际含义不符（实际是到2024-03-19 15:00:00才能计算出该行的数值）

    # 仅保留新数据部分
    df_new_factor = df_new_factor.loc[update_start_date:]
    update_end_trddt = df_new_factor.index.max()
    df_all_factor_data = pd.concat([df_local_factor, df_new_factor]).sort_index()
    df_all_factor_data.index.name = 'datetime'
    # print(df_all_factor_data)
    df_all_factor_data.reset_index(inplace=True)
    start_time = time.time()
    df_all_factor_data.to_feather(fp_factor)
    end_time = time.time()
    logger.info(f"{factor_name}因子更新完成 {fp_factor}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_factor_data)/(end_time-start_time), 2)}row/s")
    return fp_factor, factor_name, df_new_factor

def update_compound_factor_data_xmin(data_1min_path, fatacor_xmin_path, factor_name, frequency, benchmark="000852.XSHG", jobs=8, debug=False, **kwargs):
    """
    因子更新函数模板
    """
    result = _update_compound_factor_data_xmin(data_1min_path=data_1min_path, fatacor_xmin_path=fatacor_xmin_path, 
                                               factor_name=factor_name, frequency=frequency, jobs=jobs, debug=debug, benchmark=benchmark, **kwargs)

    return result

def _combine_index_related_factor(factor_post_1d_datapath, factor_name:str):
    """
    合并与指数相关的因子
        - 因为指数的分钟级数据起点不同导致计算的结果有区别
    """
    if not os.path.exists(factor_post_1d_datapath):
        os.makedirs(factor_post_1d_datapath)
    # 合并因子
    logger.info(f"开始合并{factor_name}因子")
    df_factor_000905 = pd.read_feather(os.path.join(factor_post_1d_datapath, f'{factor_name}_000905.XSHG.feather'))
    df_factor_000905.set_index(['date'], inplace=True)
    df_factor_000852 = pd.read_feather(os.path.join(factor_post_1d_datapath, f'{factor_name}_000852.XSHG.feather'))
    df_factor_000852.set_index(['date'], inplace=True)
    # 以2015年分界
    df_factor = pd.concat([
        df_factor_000905.loc[:"2015-01-01"],
        df_factor_000852.loc["2015-01-01":]
    ]).sort_index()
    df_factor.index.name = 'date'
    fp = os.path.join(factor_post_1d_datapath, f'{factor_name}.feather')
    df_factor.reset_index(inplace=True)
    start_time = time.time()
    df_factor.to_feather(fp)
    end_time = time.time()
    logger.info(f"{factor_name}因子合并完成 {fp}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_factor)/(end_time-start_time), 2)}row/s")
    return (fp, factor_name, df_factor)

def combine_index_related_factors(factor_post_1d_datapath, target_factors:list, jobs=8):
    """
    合并与指数相关的因子
        - 因为指数的分钟级数据起点不同导致计算的结果有区别
    """
    # 并行版
    if jobs > len(target_factors):
        jobs = len(target_factors)
    datas = Parallel(n_jobs=jobs)(delayed(_combine_index_related_factor)(factor_post_1d_datapath, factor_name) for factor_name in target_factors)
    return datas

    # 普通版
    # results = list()
    # if not os.path.exists(factor_post_1d_datapath):
    #     os.makedirs(factor_post_1d_datapath)
    # # 添加待合并因子
    # for factor_name in target_factors:
    #     logger.info(f"开始合并{factor_name}因子")
    #     df_factor_000905 = pd.read_feather(os.path.join(factor_post_1d_datapath, f'{factor_name}_000905.XSHG.feather'))
    #     df_factor_000905.set_index(['date'], inplace=True)
    #     df_factor_000852 = pd.read_feather(os.path.join(factor_post_1d_datapath, f'{factor_name}_000852.XSHG.feather'))
    #     df_factor_000852.set_index(['date'], inplace=True)
    #     # 以2015年分界
    #     df_factor = pd.concat([
    #         df_factor_000905.loc[:"2015-01-01"],
    #         df_factor_000852.loc["2015-01-01":]
    #     ]).sort_index()
    #     fp = os.path.join(factor_post_1d_datapath, f'{factor_name}.feather')
    #     df_factor.reset_index(inplace=True)
    #     start_time = time.time()
    #     df_factor.to_feather(fp)
    #     end_time = time.time()
    #     logger.info(f"{factor_name}因子合并完成 {fp}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_factor)/(end_time-start_time), 2)}row/s")
    #     results.append((fp, factor_name, df_factor))
    # return results