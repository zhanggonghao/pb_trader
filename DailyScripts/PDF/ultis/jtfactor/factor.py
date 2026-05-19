import numpy as np
import pandas as pd
from scipy.ndimage import shift
from numpy.linalg import LinAlgError
from datetime import datetime, timedelta
# import pandas_ta as pta # type: ignore
import ultis.jtfactor.common as jtfc
from ultis.jtfactor.common import (
    rolling_window, roll_window,
    regression, ts_regression,
    rolling_zscore
)


np.seterr(divide='ignore', invalid='ignore')


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """

    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = (
                func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
            )
        return func

    return decorate_func


def jt_VWAP(
    df,
    *factor_columns,
    weight_column='volume'
):
    """加权因子计算

    :param df: 待计算数据，以datetime或date索引，含多列相关数据。示例如下：
                         close   open
    datetime
    2023-03-01 09:31:00  13.81  13.80
    2023-03-01 09:32:00  13.79  13.81
    2023-03-01 09:33:00  13.77  13.79
    后续df若无特殊说明，其格式均与此类似。
    :type df: pd.DataFrame
    :param factor_columns: 待加权因子列，支持多参数输入。示例：'close', 'open'
    :type factor_columns: List[str]
    :param weight_column: 权重列名

    :return: 加权后因子结果，列名不变
    :return type: pd.Series, float
    """
    factor_columns = list(factor_columns)
    df[factor_columns] = df[factor_columns].multiply(df[weight_column], axis=0)
    df_sum = df.sum(numeric_only=True, axis=0)
    df_sum[factor_columns] = df_sum[factor_columns].div(df_sum[weight_column], axis=0)
    return df_sum[factor_columns].squeeze()


def jt_vwap_numpy(
    df,
    close=('close', ),
    volume='volume'
):
    close = list(close)
    price_data = df[close].values
    volume_data = df[volume].values
    weighted_prices = price_data * volume_data[:, np.newaxis]
    vwap = np.sum(weighted_prices, axis=0) / np.sum(volume_data)

    return vwap.squeeze().item()


@set_property('by_date', True)
def jt_RVOL(df, pct_change='pct_change', close='close', n=1):
    """日内收益率波动

    - 公式：RVOL_{i,t} = \sqrt{\frac{1}{K} \sum_{k=1}^{K=N} (r_k - \bar{r})^2}
    """
    if pct_change not in df.columns:
        _rets = df[close].pct_change(n)
    else:
        _rets = df[pct_change]
    _r_mean = _rets.mean()
    _rvol = np.sqrt(
        np.nanmean((_rets - _r_mean).pow(2)))
    return _rvol


@set_property('by_date', True)
def jt_RSKEW(df, pct_change='pct_change', close='close', n=1):
    """日内收益率的偏度

    - 公式：RSKEW_{i,t} = \frac{\frac{1}{K} \sum_{k=1}^{K=N} (r_k - \bar{r})^3}{RVOL_{i,t}^3}
    - NaN: 0
    """
    if pct_change not in df.columns:
        _rets = df[close].pct_change(n)
    else:
        _rets = df[pct_change]
    _r_mean = _rets.mean()
    _rvol = jt_RVOL(df, pct_change)
    _rskew = np.nanmean((_rets - _r_mean).pow(3).sum()) / np.power(_rvol, 3)
    return _rskew


@set_property('by_date', True)
def jt_RKURT(df, pct_change='pct_change', close='close', n=1):
    """日内收益率的峰度

    - 公式：RKURT_{i,t} = \frac{\frac{1}{K} \sum_{k=1}^{K=N} (r_k - \bar{r})^4}{RVOL_{i,t}^4}
    - NaN: 1
    """
    if pct_change not in df.columns:
        _rets = df[close].pct_change(n)
    else:
        _rets = df[pct_change]
    _r_mean = _rets.mean()
    _rvol = jt_RVOL(df, pct_change)
    _rkurt = np.nanmean((_rets - _r_mean).pow(4).sum()) / np.power(_rvol, 4)
    return _rkurt


@set_property('by_date', True)
def jt_VVOL(df, volume='volume'):
    return jt_RVOL(df, volume)


@set_property('by_date', True)
def jt_VSKEW(df, volume='volume'):
    """

    - NaN: 0
    """
    return jt_RSKEW(df, volume)


@set_property('by_date', True)
def jt_VKURT(df, volume='volume'):
    """

    - NaN: 1
    """
    return jt_RKURT(df, volume)


@set_property('by_date', True)
def jt_VHHI(df, volume='volume'):
    """日内HHI指数

    - 公式：VHHI_{i,t} = \sum_{k=1}^{K=48} \left( \frac{v_k}{\sum_{k=1}^{K=48} v_k} \right)^2
    - NaN: 0
    """
    _v = df[volume]
    sum_of_v = _v.sum()
    _vhhi = ((_v / sum_of_v) ** 2).sum()
    return _vhhi


def jt_NWSTD_ADJUST(df, window=20):
    """经过Newey-West标准差调整后的因子"""
    _roll_nw_std = df.rolling(window, min_periods=5).apply(jtfc.get_newey_west_adjust_std)
    return _roll_nw_std


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDRVOL(df, pct_change='pct_change', window=20, nn=False):
    """日内收益率波动的NW调整标准差

    - 公式：RVOL_{i,t} = \sqrt{\frac{1}{K} \sum_{k=1}^{K=48} (r_k - \bar{r})^2}
    - NaN: 0

    :param df: 分钟级收益率（频率高于日度）序列
    :param pct_change: 收益率列字段名，默认'pct_change'
    :param window: 计算NW调整标准差的区间长度，默认20
    :param nn: 去掉NaN值，默认不去掉

    :return: 经过NW标准差调整的日度日内收益率波动序列
    :return type:pd.Series
    """
    # _rvol = df.resample('1D').apply(jt_RVOL, pct_change=pct_change).dropna()
    # 首先计算日度数据
    _rvol = df.groupby(df.index.date).apply(jt_RVOL, pct_change=pct_change)
    # 在日度数据基础上滚动window周期取均值并计算SD
    _roll_mean = _rvol.rolling(window=window).mean()
    if nn:
        _rvol.dropna(inplace=True)
        _roll_mean.dropna(inplace=True)
    return jt_NWSTD_ADJUST(_rvol, window=window) / _roll_mean


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDRSKEW(df, pct_change='pct_change', window=20):
    """日内收益率偏度的NW调整标准差

    - NaN: 0
    """
    # FIXMEd: rskew本身存在nan值，这里dropna会有误杀的可能
    # FIXMEd: NaN值如何处理，目前均保留，满足min_periods设置即可（尽量不在计算过程中修改数据）
    # _rkurt = df.resample('1D').apply(jt_RKURT, pct_change=pct_change).dropna()
    _rskew = df.groupby(df.index.date).apply(jt_RSKEW, pct_change=pct_change)
    return jt_NWSTD_ADJUST(_rskew, window=window)


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDRKURT(df, pct_change='pct_change', window=20):
    """日内收益率峰度的NW调整标准差

    - NaN: 0
    """
    _rkurt = df.groupby(df.index.date).apply(jt_RKURT, pct_change=pct_change)
    return jt_NWSTD_ADJUST(_rkurt, window=window)


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDVVOL(df, volume='volume', window=20):
    """日内成交量波动的NW调整标准差

    - NaN: 0
    """
    _vvol = df.groupby(df.index.date).apply(jt_VVOL, volume=volume)
    _nw_vvol = jt_NWSTD_ADJUST(_vvol, window=window)
    _roll_mean = _vvol.rolling(window=window).mean()
    return _nw_vvol / _roll_mean


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDVSKEW(df, volume='volume', window=20):
    """日内成交量偏度的NW调整标准差"""
    _vskew = df.groupby(df.index.date).apply(jt_VSKEW, volume=volume)
    return jt_NWSTD_ADJUST(_vskew, window=window)


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDVKURT(df, volume='volume', window=20):
    """日内成交量峰度的NW调整标准差"""
    _vkurt = df.groupby(df.index.date).apply(jt_VKURT, volume=volume)
    return jt_NWSTD_ADJUST(_vkurt, window=window)


@set_property('by_date', False)
@set_property('window', 20)
def jt_SDVHHI(df, volume='volume', window=20):
    """日内VHHI的NW调整标准差"""
    _vhhi = df.groupby(df.index.date).apply(jt_VHHI, volume=volume)
    return jt_NWSTD_ADJUST(_vhhi, window=window)


"""
华泰证券27个日内分钟因子（部分）
"""


@set_property('by_date', True)
def jt_LATE_RET_SKEW(df, pct_change='pct_change', close='close', n=1):
    """尾盘收益率偏度

    - 尾盘：14:30以后
    - 偏度：3阶矩
    - 因子方向：-1
    """
    if df.index.name != 'datetime':
        df.set_index('datetime', inplace=True)
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    # 保留尾盘数据
    # _df = df.between_time('14:31', '15:00', include_end=True)
    _df = df.between_time('14:31', '15:00', inclusive="right")
    _ret = _df[pct_change]
    return jtfc.skewness(_ret)


@set_property('by_date', True)
def jt_DOWN_RET_VOL_PCT(df, pct_change='pct_change', close='close', n=1):
    """下行收益率波动占比

    - 推理：当天分钟级别bar中，下跌bar的收益率的波动率占当天所有bar收益率波动率的比重
    - 因子方向：1
    - fillna: 0
    """
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    _ret = np.array(df[pct_change].dropna())
    _all_std = jtfc.std(_ret)
    _ret_down = _ret[_ret < 0]
    _down_std = jtfc.std(_ret_down)
    return np.divide(_down_std, _all_std)


@set_property('by_date', True)
def jt_CORR_RET_LASTRET(df, pct_change='pct_change', close='close', n=1, lag=1):
    """前后N分钟收益率相关性

    - 推理：同样5min的收益率序列，计算1阶自相关系数
    - 因子方向：-1
    - fillna: 1 （相同的序列，相关系数为1）
    """
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    _ret = np.array(df[pct_change].dropna())
    _lag_ret = np.roll(_ret, lag)[1:]
    _ret = _ret[1:]
    return jtfc.corr(_ret, _lag_ret)


@set_property('by_date', True)
def jt_CORR_CLOSE_NEXTOPEN(df, close='close', open='open'):
    """前一分钟收盘价和后一分钟开盘价的相关性

    - 因子方向：1
    - fillna: 1 （相同的序列，相关系数为1）
    """
    _close = np.array(df[close])
    _prev_close = np.roll(_close, 1)[1:]
    _open = np.array(df[open])[1:]
    return jtfc.corr(_prev_close, _open)


def _jt_volume_pctn(df, n=2, volume='volume'):
    """前N个半小时成交量占全天成交量比重

    - 基类因子
    - 因子方向：1
    - NaN: 1
    """
    _open_time = df.index.min()  # 一般是09:31
    if n <= 4:
        _end_time = _open_time + pd.Timedelta(minutes=n * 30)
    else:
        _end_time = _open_time + pd.Timedelta(hours=3, minutes=30) + pd.Timedelta(minutes=(n - 4) * 30)
    _vol = df[volume]
    _total_vol = _vol.sum()
    _sub_vol = _vol[_vol.index < _end_time].sum()
    return np.divide(_sub_vol, _total_vol)


@set_property('by_date', True)
def jt_VOLUME_PCT2(df, **kwargs):
    return _jt_volume_pctn(df, 2, **kwargs)


@set_property('by_date', True)
def jt_VOLUME_PCT3(df, **kwargs):
    return _jt_volume_pctn(df, 3, **kwargs)


@set_property('by_date', True)
def jt_VOLUME_PCT4(df, **kwargs):
    return _jt_volume_pctn(df, 4, **kwargs)


@set_property('by_date', True)
def jt_VOLUME_PCT5(df, **kwargs):
    return _jt_volume_pctn(df, 5, **kwargs)


@set_property('by_date', True)
def jt_VOLUME_PCT6(df, **kwargs):
    return _jt_volume_pctn(df, 6, **kwargs)


@set_property('by_date', True)
def jt_VOLUME_PCT7(df, **kwargs):
    return _jt_volume_pctn(df, 7, **kwargs)


@set_property('by_date', True)
def jt_CORR_VOLUME_NTRADES(df,
                           volume='volume',
                           num_trades='num_trades'):
    """成交量与成交笔数的相关系数

    - 因子方向：-1
    """
    if num_trades not in df.columns:
        return np.NaN
    _vol = np.array(df[volume])
    _num_trades = np.array(df[num_trades])
    return jtfc.corr(_vol, _num_trades)


@set_property('by_date', True)
def jt_MOR_CORR_VOLUME_RET(df,
                           volume='volume',
                           pct_change='pct_change', close='close', n=1):
    """早盘成交量与收益率的相关系数

    - 早盘：09:31-11:30
    - 因子方向：1
    - Null值：停牌
    - fillna: 1 （相同的序列，相关系数为1）
    """
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    _df_mor = df.between_time('09:31', '11:30')
    _vol_mor = _df_mor[volume].values[1:]
    _ret_mor = _df_mor[pct_change].values[1:]
    return jtfc.corr(_vol_mor, _ret_mor)


def jt_BIGORDER_RET(df, *columns):
    """大单推动涨幅

    - 暂时无法实现，无大单数据
    """
    return


def jt_DOWN_SINGLE_AMT_PCT(df, *columns):
    """下行单笔成交金额占比

    - 暂时无法实现，无单笔成交金额数据
    """
    return


@set_property('by_date', True)
def jt_CORR_VOLUME_AMPLITUDE(df,
                             volume='volume',
                             close='close',
                             open='open'):
    """成交量与振幅的相关系数

    - 因子方向：-1
    """
    _amp = df.eval(f'{close} / {open} - 1')
    _vol = df[volume]
    return jtfc.corr(_vol, _amp)


@set_property('by_date', True)
@set_property('base_frequency', '5min')
def jt_APB(df):
    """均价偏差因子（Average Price Bias）

    - 公式：APB_{i,m} = \ln \left( \frac{\frac{1}{T} \sum_{t=1}^{T} vwap^t_{i,m}}{\frac{1}{\sum_{t=1}^{T} volu^t_{i,m}} \sum_{t=1}^{T} volu^t_{i,m} \cdot vwap^t_{i,m}} \right)
    - 要点1：APB本质是一个降频因子，将高频的信息整合后降频至低频，且原公式有两层降频：raw数据到VWAP数据降频1次，VWAP到最终因子又降频一次
    - 要点2：如果要分钟构建日度，则当前采取1min-5min-1D的思路进行构建

    - 601607.XSHG: 20081024和20081027没有分钟数据，但是有日度数据
    - fillna: 0

    :param df: 分钟级行情数据（含VWAP字段）
    """

    # 1min转5min
    #     _s_l1_vwap = df.resample('5min').apply(jt_vwap).dropna()['close']
    # _s_l1_close = df.resample('5min', closed='right', label='right').agg({'close': 'last'})['close']
    # _s_l1_vwap = df.resample('5min', closed='right', label='right').apply(jt_vwap_numpy).rename('vwap')
    # _s_l1_vwap = pd.concat([_s_l1_close, _s_l1_vwap], axis=1).ffill(axis=1)['vwap'].rename('close')
    # # vwap的NaN值以close填充
    #
    # _s_l1_vol = df.resample('5min', closed='right', label='right').agg({'volume': 'sum'})['volume']
    # # 选择正确时间段的数据（resample会额外添加非交易的时刻进来）
    # _s_l1_vol = _s_l1_vol[
    #     (_s_l1_vol.between_time('09:31', '11:31').index.tolist() + _s_l1_vol.between_time('13:01', '15:01').index.tolist())
    # ]
    # _df_l1 = pd.concat([_s_l1_vwap, _s_l1_vol], axis=1).dropna()
    # 20240325: 直接读取本地5min的K线进行计算
    _s_l1_vwap = df['vwap']
    _l2_vwap = jt_vwap_numpy(df)
    _apb = np.log(np.divide(jtfc.mean(_s_l1_vwap), _l2_vwap))
    return _apb.squeeze().item()


@set_property('by_date', True)
def jt_ARPP(df):
    """时间加权平均的股价位置（time weighed average relative price position，ARPP）

    - 公式：ARPP_{i,t} = \frac{\left( \int_{0}^{T} P_{i,t} \cdot dt - L_i \right)}{(H_i - L_i)}
    - 研报中剔除了盘中价格触及涨跌停的交易日的数据，这里暂时没有处理
    """
    _high = df['high'].max()
    _low = df['low'].min()
    _twap = df['close'].mean()
    return (_twap - _low) / (_high - _low + 1e-5)


@set_property('by_date', False)
def _jt_arpp_n_m(df, n=1, m=20):
    """不同窗口滚动的ARPP均值"""
    n_mins = n * 240
    _r_df = df.rolling(n_mins, min_periods=240).agg({
        'high': 'max', 'low': 'min', 'close': 'mean'
    })
    _s_arpp = _r_df.eval('(close - low) / (high - low + 1e-5)')
    _s_arpp = _s_arpp.groupby(_s_arpp.index.date).last()
    _arpp_n_m = _s_arpp.rolling(m, min_periods=n).mean()
    return _arpp_n_m


@set_property('by_date', False)
@set_property('window', 20)  # window=20+1-1
def jt_ARPP_1_20(df):
    return _jt_arpp_n_m(df, 1, 20)


@set_property('by_date', False)
@set_property('window', 24)  # window=20+5-1
def jt_ARPP_5_20(df):
    return _jt_arpp_n_m(df, 5, 20)


@set_property('by_date', False)
@set_property('window', 39)
def jt_ARPP_20_20(df):
    return _jt_arpp_n_m(df, 20, 20)


@set_property('by_date', False)
def _jt_open_min_ret(df, n=15, pct_change='pct_change', close='close'):
    """开盘后N分钟收益率"""
    # 注解：这里最好不要提前输入pct_change，因为根据n的不同需要不同范围的pct_change，每次重新计算最好
    # if pct_change not in df.columns:
    df[pct_change] = df[close].pct_change(n)
    _time = (pd.Timestamp("09:30") + pd.Timedelta(minutes=n)).time()
    _s_ret = df[df.index.time == _time][pct_change]
    # 将索引转化为日度
    _s_ret = _s_ret.groupby(_s_ret.index.date).last()
    return _s_ret


@set_property('by_date', False)
@set_property('window', 1)
def jt_OPEN_5MIN_RET(df, **kwargs):
    """开盘5分钟收益率"""
    return _jt_open_min_ret(df, 5, **kwargs)


@set_property('by_date', False)
@set_property('window', 1)
def jt_OPEN_15MIN_RET(df, **kwargs):
    """开盘15分钟收益率"""
    return _jt_open_min_ret(df, 15, **kwargs)


@set_property('by_date', False)
@set_property('window', 1)
def jt_OPEN_30MIN_RET(df, **kwargs):
    """开盘30分钟收益率"""
    return _jt_open_min_ret(df, 30, **kwargs)


@set_property('by_date', False)  # 不by_date会更快一点
def _jt_close_min_ret(df, n=15, pct_change='pct_change', close='close'):
    """收盘前N分钟收益率"""
    # 注解：这里最好不要提前输入pct_change，因为根据n的不同需要不同范围的pct_change，每次重新计算最好
    # if pct_change not in df.columns:
    df[pct_change] = df[close].pct_change(n)
    # _time = (pd.Timestamp("15:00") - pd.Timedelta(minutes=n)).time()
    _s_ret = df[df.index.time == pd.Timestamp("15:00").time()][pct_change]
    # 将索引转化为日度
    _s_ret = _s_ret.groupby(_s_ret.index.date).last()
    return _s_ret


@set_property('by_date', False)
def jt_CLOSE_5MIN_RET(df, **kwargs):
    """收盘前5分钟收益率"""
    return _jt_close_min_ret(df, 5, **kwargs)


@set_property('by_date', False)
def jt_CLOSE_15MIN_RET(df, **kwargs):
    """收盘前15分钟收益率"""
    return _jt_close_min_ret(df, 15, **kwargs)


@set_property('by_date', False)
def jt_CLOSE_30MIN_RET(df, **kwargs):
    """收盘30分钟收益率"""
    return _jt_close_min_ret(df, 30, **kwargs)


def _filter_matrix_rows_by_date_ends(data: np.ndarray):
    """给定矩阵形式的rolling表，提取符合条件的连续行"""
    last_dates_minutes = np.array([pd.to_datetime(date).time() for date in data[:, -1]])
    date_ends = last_dates_minutes == pd.Timestamp("15:00").time()
    return date_ends


def _filter_rows_by_date_end(data):
    """给定array形式的连续列表，提取符合条件的单个行

    :return: 返回True-False的连续列表，其中True为当日结束
    :return type: list[bool]
    """
    date_ends = pd.to_datetime(data).time == pd.Timestamp('15:00').time()
    return date_ends


def _jt_rolling_cr(df, window, side, naive_return='pct_change', benchmark_return='bm_pct_change', keepnan=True):
    """相关系数计算"""
    if naive_return not in df.columns:
        df[naive_return] = df['close'].pct_change()
    s_idx = df.index.to_numpy()
    s_naive_ret = df[naive_return].to_numpy()
    s_bm_ret = df[benchmark_return].to_numpy()

    r_idx = rolling_window(s_idx, window)
    valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
    r_idx_valid = r_idx[valid_rows]
    s_idx_valid = r_idx_valid[:, -1]
    r_naive = rolling_window(s_naive_ret, window)[valid_rows]
    r_bm = rolling_window(s_bm_ret, window)[valid_rows]
    if side == 'down':
        r_naive_dd = np.where(r_bm < 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm < 0, r_bm, np.NaN)
    elif side == 'up':
        r_naive_dd = np.where(r_bm > 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm > 0, r_bm, np.NaN)
    else:
        raise ValueError("side参数只支持down和up。")

    n_rows = r_naive.shape[0]
    coefs = []
    for i in range(n_rows):
        s_naive_dd = r_naive_dd[i, :]
        s_benchmark_dd = r_benchmark_dd[i, :]

        if keepnan:
            nn_idx = np.where(~np.isnan(s_naive_dd))
            s_naive_dd = s_naive_dd[nn_idx]
            s_benchmark_dd = s_benchmark_dd[nn_idx]
        else:
            s_naive_dd = s_naive_dd[~np.isnan(s_naive_dd)]
            s_benchmark_dd = s_benchmark_dd[~np.isnan(s_benchmark_dd)]

        if len(s_naive_dd) == 0:
            coef = np.nan
        else:
            coef = np.corrcoef(s_naive_dd, s_benchmark_dd)[0, 1]  # np原生更快，且两者结果一样（np.corrcoef返回的也是Pearson Correlation）
        coefs.append(coef)
    # 如果只有一个值，说明函数是groupby运行，直接返回最后一个值
    # if len(coefs) == 1:
    #     return coefs[-1]
    # else:
    #     # s_coefs = pd.Series(coefs, index=s_idx[-len(coefs):])
    #     # s_idx_date = pd.to_datetime(s_idx_valid.date)
    s_coefs = pd.Series(coefs, index=s_idx_valid)
    return s_coefs


@set_property('by_date', False)
@set_property('base_frequency', '1min')
def _jt_ddncr(df, window, **kwargs):
    return _jt_rolling_cr(df, window, side='down', **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
def _jt_uupcr(df, window, **kwargs):
    return _jt_rolling_cr(df, window, side='up', **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
def jt_DDNCR240(df, **kwargs):
    return _jt_ddncr(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)  # 1 = 480 / 240 - 1
def jt_DDNCR480(df, **kwargs):
    return _jt_ddncr(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)
def jt_DDNCR720(df, **kwargs):
    return _jt_ddncr(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNCR960(df, **kwargs):
    return _jt_ddncr(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNCR1200(df, **kwargs):
    return _jt_ddncr(df, window=1200, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
def jt_UUPCR240(df, **kwargs):
    return _jt_uupcr(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)  # 1 = 480 / 240 - 1
def jt_UUPCR480(df, **kwargs):
    return _jt_uupcr(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)
def jt_UUPCR720(df, **kwargs):
    return _jt_uupcr(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_UUPCR960(df, **kwargs):
    return _jt_uupcr(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_UUPCR1200(df, **kwargs):
    return _jt_uupcr(df, window=1200, **kwargs)


def _jt_rolling_sr(df, window, side, naive_return='pct_change', benchmark_return='bm_pct_change', keepnan=True):
    """波动计算"""
    if naive_return not in df.columns:
        df[naive_return] = df['close'].pct_change()
    s_idx = df.index.to_numpy()
    s_naive_ret = df[naive_return].to_numpy()
    s_bm_ret = df[benchmark_return].to_numpy()

    r_idx = rolling_window(s_idx, window)
    valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
    r_idx_valid = r_idx[valid_rows]
    s_idx_valid = r_idx_valid[:, -1]
    r_naive = rolling_window(s_naive_ret, window)[valid_rows]
    r_bm = rolling_window(s_bm_ret, window)[valid_rows]
    if side == 'down':
        r_naive_dd = np.where(r_bm < 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm < 0, r_bm, np.NaN)
    elif side == 'up':
        r_naive_dd = np.where(r_bm > 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm > 0, r_bm, np.NaN)
    else:
        raise ValueError("side参数只支持down和up。")

    std_naive = np.nanstd(r_naive_dd, axis=1)
    std_benchmark = np.nanstd(r_benchmark_dd, axis=1)

    sr = std_naive / std_benchmark
    # 如果只有一个值，说明函数是groupby运行，直接返回最后一个值
    # if len(sr) == 1:
    #     return sr[-1]
    # else:
    s_sr = pd.Series(sr, index=s_idx_valid)
    return s_sr


def _jt_ddnsr(df, window, **kwargs):
    return _jt_rolling_sr(df, window, side='down', **kwargs)


def _jt_uupsr(df, window, **kwargs):
    return _jt_rolling_sr(df, window, side='up', **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
def jt_DDNSR240(df, **kwargs):
    return _jt_ddnsr(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)  # 1 = 480 / 240 - 1
def jt_DDNSR480(df, **kwargs):
    return _jt_ddnsr(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)
def jt_DDNSR720(df, **kwargs):
    return _jt_ddnsr(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNSR960(df, **kwargs):
    return _jt_ddnsr(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNSR1200(df, **kwargs):
    return _jt_ddnsr(df, window=1200, **kwargs)


@set_property('by_date', False)
def jt_UUPSR240(df, **kwargs):
    return _jt_uupsr(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)  # 1 = 480 / 240 - 1
def jt_UUPSR480(df, **kwargs):
    return _jt_uupsr(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)
def jt_UUPSR720(df, **kwargs):
    return _jt_uupsr(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_UUPSR960(df, **kwargs):
    return _jt_uupsr(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_UUPSR1200(df, **kwargs):
    return _jt_uupsr(df, window=1200, **kwargs)


def _jt_rolling_bt(df, window, side, naive_return='pct_change', benchmark_return='bm_pct_change', keepnan=True):
    """波动计算"""
    if naive_return not in df.columns:
        df[naive_return] = df['close'].pct_change()
    s_idx = df.index.to_numpy()
    s_naive_ret = df[naive_return].to_numpy()
    s_bm_ret = df[benchmark_return].to_numpy()

    r_idx = rolling_window(s_idx, window)
    valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
    r_idx_valid = r_idx[valid_rows]
    s_idx_valid = r_idx_valid[:, -1]
    r_naive = rolling_window(s_naive_ret, window)[valid_rows]
    r_bm = rolling_window(s_bm_ret, window)[valid_rows]
    if side == 'down':
        r_naive_dd = np.where(r_bm < 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm < 0, r_bm, np.NaN)
    elif side == 'up':
        r_naive_dd = np.where(r_bm > 0, r_naive, np.NaN)
        r_benchmark_dd = np.where(r_bm > 0, r_bm, np.NaN)
    else:
        raise ValueError("side参数只支持down和up。")

    n_rows = r_naive.shape[0]
    betas = []
    for i in range(n_rows):
        s_naive_dd = r_naive_dd[i, :]
        s_benchmark_dd = r_benchmark_dd[i, :]

        if keepnan:
            nn_idx = np.where(~np.isnan(s_naive_dd))
            s_naive_dd = s_naive_dd[nn_idx]
            s_benchmark_dd = s_benchmark_dd[nn_idx]
        else:
            s_naive_dd = s_naive_dd[~np.isnan(s_naive_dd)]
            s_benchmark_dd = s_benchmark_dd[~np.isnan(s_benchmark_dd)]

        if len(s_naive_dd) == 0:
            beta = np.nan
        else:
            beta = np.polyfit(s_benchmark_dd, s_naive_dd, 1)[0]
        betas.append(beta)
    # 如果只有一个值，说明函数是groupby运行，直接返回最后一个值
    # if len(betas) == 1:
    #     return betas[-1]
    # else:
    #     # s_idx_date = pd.to_datetime(s_idx_valid.date)
    s_betas = pd.Series(betas, index=s_idx_valid)
    return s_betas


def _jt_ddnbt(df, window, **kwargs):
    return _jt_rolling_bt(df, window, side='down', **kwargs)


def _jt_uupbt(df, window, **kwargs):
    return _jt_rolling_bt(df, window, side='up', **kwargs)


@set_property('by_date', False)
def jt_DDNBT240(df, **kwargs):
    return _jt_ddnbt(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)
def jt_DDNBT480(df, **kwargs):
    return _jt_ddnbt(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)
def jt_DDNBT720(df, **kwargs):
    return _jt_ddnbt(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNBT960(df, **kwargs):
    return _jt_ddnbt(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)
def jt_DDNBT1200(df, **kwargs):
    return _jt_ddnbt(df, window=1200, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 2)  # 1 = 480 / 240 - 1
def jt_UUPBT240(df, **kwargs):
    return _jt_uupbt(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)  # 1 = 480 / 240 - 1
def jt_UUPBT480(df, **kwargs):
    return _jt_uupbt(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)  # 1 = 480 / 240 - 1
def jt_UUPBT720(df, **kwargs):
    return _jt_uupbt(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 4)  # 1 = 480 / 240 - 1
def jt_UUPBT960(df, **kwargs):
    return _jt_uupbt(df, window=960, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '1min')
@set_property('window', 5)  # 1 = 480 / 240 - 1
def jt_UUPBT1200(df, **kwargs):
    return _jt_uupbt(df, window=1200, **kwargs)


def _jt_tobt(df, window, naive_return='pct_change', benchmark_return='bm_pct_change', turnover='total_turnover'):
    """超额流动（Liquidity-turnover beta）因子计算

    - 20240327：原因子构建采用的是换手率，这里直接使用total_turnover代替，因为在分钟频率上市值并不会有明显的变化，
               同时，为了避免系数太小，这里对total_turnover进行单位变化处理
    """
    # 超额流动涉及5阶滞后自回归，所以这里要额外请求数据
    # window += 5
    if naive_return not in df.columns:
        df[naive_return] = df['close'].pct_change()
    s_idx = df.index.to_numpy()
    s_naive_ret = df[naive_return].to_numpy()
    s_bm_ret = df[benchmark_return].to_numpy()
    s_turnover = df[turnover].to_numpy()

    r_idx = rolling_window(s_idx, window + 5)
    valid_rows = _filter_matrix_rows_by_date_ends(r_idx)

    r_idx_valid = r_idx[valid_rows]
    s_idx_valid = r_idx_valid[:, -1]
    r_naive_abs = np.abs(rolling_window(s_naive_ret, window + 5))[valid_rows]
    r_benchmark_abs = np.abs(rolling_window(s_bm_ret, window + 5))[valid_rows]
    r_turnover = rolling_window(s_turnover, window + 5)[valid_rows]
    # TODO: 后续还是需要使用市值来约束单位，简单除以1e7可能存在潜在偏差
    r_turnover /= 1e7
    # 市场组合收益的绝对值的5阶滞后
    r_benchmark_1 = np.apply_along_axis(lambda i: shift(i, 1, cval=np.NaN), axis=1, arr=r_benchmark_abs)
    r_benchmark_2 = np.apply_along_axis(lambda i: shift(i, 2, cval=np.NaN), axis=1, arr=r_benchmark_abs)
    r_benchmark_3 = np.apply_along_axis(lambda i: shift(i, 3, cval=np.NaN), axis=1, arr=r_benchmark_abs)
    r_benchmark_4 = np.apply_along_axis(lambda i: shift(i, 4, cval=np.NaN), axis=1, arr=r_benchmark_abs)
    r_benchmark_5 = np.apply_along_axis(lambda i: shift(i, 5, cval=np.NaN), axis=1, arr=r_benchmark_abs)
    # 个股收益的绝对值的5阶滞后
    r_ret_1 = np.apply_along_axis(lambda i: shift(i, 1, cval=np.NaN), axis=1, arr=r_naive_abs)
    r_ret_2 = np.apply_along_axis(lambda i: shift(i, 2, cval=np.NaN), axis=1, arr=r_naive_abs)
    r_ret_3 = np.apply_along_axis(lambda i: shift(i, 3, cval=np.NaN), axis=1, arr=r_naive_abs)
    r_ret_4 = np.apply_along_axis(lambda i: shift(i, 4, cval=np.NaN), axis=1, arr=r_naive_abs)
    r_ret_5 = np.apply_along_axis(lambda i: shift(i, 5, cval=np.NaN), axis=1, arr=r_naive_abs)

    n_rows = r_naive_abs.shape[0]
    betas = []
    for i in range(n_rows):
        s_ret = r_naive_abs[i, :]
        s_turnover = r_turnover[i, :]
        s_ret_1 = r_ret_1[i, :]
        s_ret_2 = r_ret_2[i, :]
        s_ret_3 = r_ret_3[i, :]
        s_ret_4 = r_ret_4[i, :]
        s_ret_5 = r_ret_5[i, :]
        s_benchmark_1 = r_benchmark_1[i, :]
        s_benchmark_2 = r_benchmark_2[i, :]
        s_benchmark_3 = r_benchmark_3[i, :]
        s_benchmark_4 = r_benchmark_4[i, :]
        s_benchmark_5 = r_benchmark_5[i, :]
        X = np.column_stack([
            np.ones(len(s_ret)),
            np.abs(s_turnover),
            np.abs(s_ret_1), np.abs(s_ret_2),
            np.abs(s_ret_3), np.abs(s_ret_4), np.abs(s_ret_5),
            np.abs(s_benchmark_1), np.abs(s_benchmark_2), np.abs(s_benchmark_3),
            np.abs(s_benchmark_4), np.abs(s_benchmark_5)
        ])[- window:]  # 与V1相同，因为输入stack的是一维向量，stack后变为505*12维的数组
        y = np.array(np.abs(s_ret))[-window:]  # s_ret是向量，所以直接[-500:]读取
        try:
            beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)[1]  # 第0项是常数项，第1项是所求系数
        except LinAlgError:
            beta = np.NaN
        betas.append(beta)
    # 如果只有一个值，说明函数是groupby运行，直接返回最后一个值
    # if len(betas) == 1:
    #     return betas[-1]
    # else:
    #     # s_idx_date = pd.to_datetime(s_idx_valid.date)
    s_betas = pd.Series(betas, index=s_idx_valid)
    return s_betas


@set_property('by_date', False)
@set_property('base_frequency', '5min')
@set_property('window', 6)  # 4 = 240 * 5 / 240 - 1
def jt_TOBT240(df, **kwargs):
    return _jt_tobt(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '5min')
@set_property('window', 11)  #
def jt_TOBT480(df, **kwargs):
    return _jt_tobt(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '5min')
@set_property('window', 16)
def jt_TOBT720(df, **kwargs):
    return _jt_tobt(df, window=720, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', '5min')
@set_property('window', 21)
def jt_TOBT960(df, **kwargs):
    return _jt_tobt(df, window=960, **kwargs)


def _signed_series(series, initial: int = None):
    """Returns a Signed Series/DataFrame with or without an initial value

    Default Example:
    series = Series([3, 2, 2, 1, 1, 5, 6, 6, 7, 5])
    and returns:
    sign = Series([NaN, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0])
    """
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


@set_property('by_date', True)
def _jt_npvi(df, side, pct_change='pct_change', volume='volume', close='close', n=1):
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    _roc = df[pct_change]
    _volume = df[volume]

    _signed_vol = _signed_series(_volume, 1)

    if side == 1:
        _vi = _signed_vol[_signed_vol > 0].abs() * _roc
    else:
        assert side == -1, "仅支持side==1或side==-1"
        _vi = _signed_vol[_signed_vol < 0].abs() * _roc
    _vi.fillna(0, inplace=True)
    _vi.iloc[0] = 100
    # _vi = _vi.cumsum()
    # return _vi.values[-1]
    return _vi.sum()


# @set_property('by_date', False)
# def _jt_npvi_cont(df, side, pct_change='pct_change', volume='volume', close='close', n=1):
#     if pct_change not in df.columns:
#         df[pct_change] = df[close].pct_change(n)
#     _roc = df[pct_change]
#     _volume = df[volume]
#
#     _signed_vol = _signed_series(_volume, 1)
#
#     if side == 1:
#         _vi = _signed_vol[_signed_vol > 0].abs() * _roc
#     else:
#         assert side == -1, "仅支持side==1或side==-1"
#         _vi = _signed_vol[_signed_vol < 0].abs() * _roc
#     _vi.fillna(0, inplace=True)
#     _vi.iloc[0] = 100
#     _vi = _vi.cumsum()
#
#     # _vi_r = roll_window(vi, window=window)
#     # return _vi.values[-1]
#     return _vi.sum()


@set_property('by_date', True)
@set_property('base_frequency', '1min')
def jt_PVI(df, **kwargs):
    """（日内）正成交量指数

    - 20240328：目前仅计算日内正累积正成交量指数，后续有需要再将计算时间延长（例如延长至5日共1200根bar）
    """
    return _jt_npvi(df, side=1, **kwargs)


@set_property('by_date', True)
@set_property('base_frequency', '1min')
def jt_NVI(df, **kwargs):
    """（日内）正成交量指数

    - 20240328：目前仅计算日内正累积正成交量指数，后续有需要再将计算时间延长（例如延长至5日共1200根bar）
    """
    return _jt_npvi(df, side=-1, **kwargs)


# def jt_NVI5(df, **kwargs):
#     return _jt_npvi(df, side=-1, **kwargs)


def _jt_rsrs(df, window=48, high='high', low='low', date_end=True):
    """RSRS阻力支撑相对强度

    计算方法：N时刻最高价和最低价的线性回归系数
    """
    _betas = np.array(ts_regression(df[high].values, df[low].values, window=window))
    _idx = df.index[window-1:]

    if date_end:
        # 如果以日度返回数据，则提取每日最后一根bar的结果作为当日的因子值
        valid_idx_loc = _filter_rows_by_date_end(_idx)
        _betas = _betas[valid_idx_loc]
        _idx = _idx[valid_idx_loc]
    # 20240402：分钟转日频的操作在函数外统一进行
    # return pd.Series(_betas_dt_end[:, 0], index=pd.to_datetime(_idx_dt_end.date))
    return pd.Series(_betas[:, 0], index=_idx) - 1


# 20240402：48和96频率太短，容易缺少足够数据用于回归
@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)  # 4 = 240 * 5 / 240 - 1
def jt_RSRS240(df, **kwargs):
    return _jt_rsrs(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 10)  # 9 = 480 * 5 / 240 - 1
def jt_RSRS480(df, **kwargs):
    return _jt_rsrs(df, window=480, **kwargs)


def _jt_rsrs_zscore(df, window1, window2, **kwargs):
    _rsrs = _jt_rsrs(df, window=window1, date_end=False, **kwargs)
    _idx = _rsrs.index[window2-1:]
    _rsrs_zscore = rolling_zscore(_rsrs, window=window2)
    valid_idx_loc = _filter_rows_by_date_end(_idx)
    _rsrs_zscore = _rsrs_zscore[valid_idx_loc]
    _idx = _idx[valid_idx_loc]
    return pd.Series(_rsrs_zscore, index=_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 15)  # 14 = (240 * 5 / 240  + 480 * 5 / 240) - 1
def jt_RSRS_ZSCORE240(df, **kwargs):
    return _jt_rsrs_zscore(df, 240, 480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 20)  # 19 = (480 * 5 / 240  + 480 * 5 / 240) - 1
def jt_RSRS_ZSCORE480(df, **kwargs):
    return _jt_rsrs_zscore(df, 480, 480, **kwargs)


def _jt_dvrat(df, window=480, naive_return='pct_change', benchmark_return='bm_pct_change', date_end=True, close='close'):
    if naive_return not in df.columns:
        df[naive_return] = df[close].pct_change(1)

    q = 10
    m = q * (window - q + 1) * (1 - q / window)

    s_idx = (df[naive_return] - df[benchmark_return]).index.values
    r_idx = roll_window(s_idx, window=window)

    s_ex_ret = (df[naive_return] - df[benchmark_return]).values
    r_ex_ret_q = roll_window(s_ex_ret, window=q)
    r_ex_ret_w = roll_window(s_ex_ret, window=window)
    s_sigma = np.power(np.nansum(r_ex_ret_q, axis=1), 2)
    s_sigma_q = np.nansum(roll_window(s_sigma, window=window - q + 1), axis=1) / m

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]

        r_ex_ret_w = r_ex_ret_w[valid_rows]
        s_sigma_q = s_sigma_q[valid_rows]

    s_sigma_s = np.nanvar(r_ex_ret_w, axis=1)
    s_dvrat = s_sigma_q / s_sigma_s - 1

    s_idx = r_idx[:, -1]
    return pd.Series(s_dvrat, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 10)
def jt_DVRAT480(df, **kwargs):
    return _jt_dvrat(df, window=480, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 20)
def jt_DVRAT960(df, **kwargs):
    return _jt_dvrat(df, window=960, **kwargs)


def _jt_obv(df, window=240, pct_change='pct_change', volume='volume', date_end=True, close='close', n=1):
    if pct_change not in df.columns:
        df[pct_change] = df[close].pct_change(n)
    r_pct_change = roll_window(df[pct_change].values, window)
    r_volume = roll_window(df[volume].values, window)
    r_idx = roll_window(df.index.values, window)

    r_sgn = np.sign(r_pct_change)  # r_pct_change / np.abs(r_pct_change)
    r_sgn = np.where(np.isnan(r_sgn), 0, r_sgn)
    r_obv = np.nansum(r_sgn * r_volume, axis=1)

    if date_end:
        valid_idx = _filter_matrix_rows_by_date_ends(r_idx)
        valid_values = r_obv[valid_idx]
        valid_idx = r_idx[valid_idx][:, -1]
        return pd.Series(valid_values, index=valid_idx)


@set_property('by_date', False)
def jt_OBV240(df):
    return _jt_obv(df, window=240)


@set_property('base_frequency', "5min")
def _jt_tr(df, window=48, high='high', low='low', close='close', date_end=True, ratio=True):
    s_idx = df.index.values
    s_high = df[high].values
    s_low = df[low].values
    s_close = df[close].values
    s_prev_close = df[close].shift(1).values
    hml = s_high - s_low
    hmpc = np.abs(s_high - s_prev_close)
    lmpc = np.abs(s_low - s_prev_close)

    r_idx = roll_window(s_idx, window=window)
    r_hml = roll_window(hml, window=window)
    r_hmpc = roll_window(hmpc, window=window)
    r_lmpc = roll_window(lmpc, window=window)
    r_close = roll_window(s_close, window=window)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_hml = r_hml[valid_rows]
        r_hmpc = r_hmpc[valid_rows]
        r_lmpc = r_lmpc[valid_rows]
        r_close = r_close[valid_rows]
        
    r_stacked = np.stack((r_hml, r_hmpc, r_lmpc))
    r_tr = np.max(r_stacked, axis=0)

    s_tr = np.nansum(r_tr, axis=1)
    if ratio:
        # 如果要去量纲，则除以移动收盘价的均值
        s_tr /= np.nanmean(r_close, axis=1)
    s_idx = r_idx[:, -1]
    return pd.Series(s_tr, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_TR48(df, **kwargs):
    return _jt_tr(df, window=48, date_end=True, **kwargs)


def _jt_atr(df, window1=5, window2=48, **kwargs):
    """平均真实振幅

    :param df: 数据，以日期为索引，以成分为列，每个df是一只个股
    :param window1: ATR计算周期，频率：日度
    :param window2: TR计算周期，默认频率：5min
    """
    _tr = _jt_tr(df, window=window2, date_end=True, **kwargs)
    s_idx = _tr.index.values
    s_tr = _tr.values
    r_idx = roll_window(s_idx, window=window1)
    r_tr = roll_window(s_tr, window=window1)

    s_atr = np.nanmean(r_tr, axis=1)
    s_idx_end = r_idx[:, -1]
    return pd.Series(s_atr, index=s_idx_end)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 20)
def jt_ATR_1_20(df, **kwargs):
    return _jt_atr(df, window1=20, window2=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 25)
def jt_ATR_5_20(df, **kwargs):
    return _jt_atr(df, window1=20, window2=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 40)
def jt_ATR_20_20(df, **kwargs):
    return _jt_atr(df, window1=20, window2=960, **kwargs)


def _jt_vr(df, window=48, close='close', volume='volume', date_end=True):
    """成交量比率（Volume Ratio）

    - 已去量纲
    """
    s_idx = df[volume].index.values
    s_vol = df[volume].values
    # s_close = df[close].values
    # s_prev_close = df[close].shift(1).values
    s_diff_close = df[close].diff().values

    r_idx = roll_window(s_idx, window=window)
    r_vol = roll_window(s_vol, window=window)
    r_close = roll_window(s_diff_close, window=window)
    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_vol = r_vol[valid_rows]
        r_close = r_close[valid_rows]

    r_av = np.where(r_close > 0, r_vol, 0)
    r_bv = np.where(r_close < 0, r_vol, 0)
    r_cv = np.where(r_close == 0, r_vol, 0)

    s_av = np.nansum(r_av, axis=1)
    s_bv = np.nansum(r_bv, axis=1)
    s_cv = np.nansum(r_cv, axis=1)

    s_vr = (s_av + s_cv / 2) / (s_bv + s_cv / 2 + 1)
    s_idx = r_idx[:, -1]
    return pd.Series(s_vr, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_VR48(df, **kwargs):
    return _jt_vr(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_VR240(df, **kwargs):
    return _jt_vr(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 10)
def jt_VR480(df, **kwargs):
    return _jt_vr(df, window=480, **kwargs)


@set_property('base_frequency', "5min")
def _jt_acd(df, window=48, close='close', high='high', low='low', date_end=True):
    
    s_idx = df.index.values
    s_close = df[close].values
    s_prev_close = df[close].shift(1).values
    s_high = df[high].values
    s_low = df[low].values

    r_idx = roll_window(s_idx, window)
    r_close = roll_window(s_close, window)
    r_prev_close = roll_window(s_prev_close, window)
    r_high = roll_window(s_high, window)
    r_low = roll_window(s_low, window)

    if date_end:
        # 如果是只要返回当日结束的数据，则提前提取有效行，从而减少运算量
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)

        r_idx = r_idx[valid_rows]
        r_close = r_close[valid_rows]
        r_prev_close = r_prev_close[valid_rows]
        r_high = r_high[valid_rows]
        r_low = r_low[valid_rows]

    r_min_low_prev_close = np.minimum(r_low, r_prev_close)
    r_max_high_prev_close = np.maximum(r_high, r_prev_close)
    r_buy = r_close - r_min_low_prev_close
    r_sell = r_close - r_max_high_prev_close

    s_acd = np.nansum(r_buy + r_sell, axis=1)
    s_mean_close = np.nanmean(r_close, axis=1)
    s_acd = s_acd / s_mean_close
    s_idx = r_idx[:, -1]

    return pd.Series(s_acd, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_ACD48(df, **kwargs):
    return _jt_acd(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_ACD240(df, **kwargs):
    return _jt_acd(df, window=240, **kwargs)


@set_property('base_frequency', "5min")
def _jt_adtm(df, window=48, open='open', close='close', high='high', low='low', date_end=True):

    s_idx = df.index.values
    s_open = df[open].values
    s_prev_open = df[open].shift(1).values

    s_close = df[close].values
    s_high = df[high].values
    s_low = df[low].values

    s_dtm = np.where(s_open <= s_prev_open, 0,
                     np.maximum(s_high - s_open,
                                s_open - s_prev_open))
    s_dbm = np.where(s_open >= s_prev_open, 0,
                     np.maximum(s_open - s_low,
                                s_open - s_prev_open))

    r_idx = roll_window(s_idx, window=window)
    r_dtm = roll_window(s_dtm, window=window)
    r_dbm = roll_window(s_dbm, window=window)
    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_dtm = r_dtm[valid_rows]
        r_dbm = r_dbm[valid_rows]

    s_idx = r_idx[:, -1]
    s_stm = np.nansum(r_dtm, axis=1)
    s_sbm = np.nansum(r_dbm, axis=1)
    s_adtm = np.where(s_stm > s_sbm, (s_stm - s_sbm) / s_stm,
                      np.where(s_stm == s_sbm, 0, (s_stm - s_sbm) / s_sbm))

    return pd.Series(s_adtm, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_ADTM48(df, **kwargs):
    return _jt_adtm(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_ADTM240(df, **kwargs):
    return _jt_adtm(df, window=240, **kwargs)


@set_property('base_frequency', "5min")
def _jt_wr(df, window=48, close='close', high='high', low='low', date_end=True):
    """威廉指数

    计算公式：
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    """
    s_idx = df.index.values
    s_close = df[close].values
    s_high = df[high].values
    s_low = df[low].values

    r_idx = roll_window(s_idx, window=window)
    r_close = roll_window(s_close, window=window)
    r_high = roll_window(s_high, window=window)
    r_low = roll_window(s_low, window=window)
    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_close = r_close[valid_rows]
        r_high = r_high[valid_rows]
        r_low = r_low[valid_rows]

    s_idx = r_idx[:, -1]
    s_close = r_close[:, -1]
    s_hhv = np.nanmax(r_high, axis=1)
    s_llv = np.nanmin(r_low, axis=1)
    s_wr = (s_hhv - s_close) / (s_hhv - s_llv + 0.00001) * 100  # +0.00001防止停牌产生无穷大数据
    return pd.Series(s_wr, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_WR48(df, **kwargs):
    """WR48
    - 本质就是当天收盘价举例高低点的位置
    """
    return _jt_wr(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_WR240(df, **kwargs):
    return _jt_wr(df, window=240, **kwargs)


@set_property('base_frequency', "5min")
def _jt_adx(df, window1=10, window2=240, close='close', high='high', low='low', date_end=True):
    s_idx = df.index.values
    # s_high = df[high].values
    # s_low = df[low].values
    # s_close = df[close].values
    # s_prev_close = df[close].shift(1).values
    s_high_diff = df[high].diff().values
    s_low_diff = df[low].diff().values * (-1)  # 原因子构造如此，即使用prev_low-low，和diff(-1)有区别

    r_idx = roll_window(s_idx, window=window2)
    r_high_diff = roll_window(s_high_diff, window=window2)
    r_low_diff = roll_window(s_low_diff, window=window2)

    r_dmp = np.where((r_high_diff > 0) & (r_high_diff > r_low_diff), r_high_diff, 0)
    r_dmm = np.where((r_low_diff > 0) & (r_low_diff > r_high_diff), r_low_diff, 0)
    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_dmp = r_dmp[valid_rows]
        r_dmm = r_dmm[valid_rows]

    _s_tr = _jt_tr(df, window=window2, date_end=date_end).values
    s_idx = r_idx[:, -1]
    s_dmp = np.nansum(r_dmp, axis=1) * 100 / _s_tr
    s_dmm = np.nansum(r_dmm, axis=1) * 100 / _s_tr

    s_adx = np.nanmean(roll_window(np.abs(s_dmm - s_dmp) / (s_dmm + s_dmp) * 100,
                                   window=window1), axis=1)
    s_idx = roll_window(s_idx, window=window1)[:, -1]
    return pd.Series(s_adx, index=s_idx)


def _jt_asi(df, window=48, close='close', open='open', high='high', low='low', date_end=True, ratio=True):
    s_idx = df.index.values
    s_open = df[open].values
    s_prev_open = df[open].shift(1).values
    s_high = df[high].values
    s_low = df[low].values
    s_prev_low = df[low].shift(1).values
    s_close = df[close].values
    s_prev_close = df[close].shift(1).values
    # s_high_diff = df[high].diff().values
    s_close_diff = df[close].diff().values

    _aa = np.abs(s_high - s_prev_close)
    _bb = np.abs(s_low - s_prev_close)
    _cc = np.abs(s_high - s_prev_low)
    _dd = np.abs(s_prev_close - s_prev_open)
    _ee = s_close_diff
    _ff = s_close - s_open
    _gg = s_prev_close - s_prev_open
    _rr = np.where(
        (_aa > _bb) & (_aa > _cc),
        _aa + _bb / 2 + _dd / 4,
        np.where(
            (_bb > _cc) & (_bb > _aa),
            _bb + _aa / 2 + _dd / 4,
            _cc + _dd / 4
        )
    )
    _xx = (_ee + _ff / 2 + _gg)
    _kk = np.maximum(_aa, _bb)
    _si = 50 * (_xx / (_rr + 1e-5)) * (_kk / 3)

    r_idx = roll_window(s_idx, window=window)
    r_si = roll_window(_si, window=window)
    r_close = roll_window(s_close, window=window)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_si = r_si[valid_rows]
        r_close = r_close[valid_rows]

    s_idx = r_idx[:, -1]
    s_asi = np.nansum(r_si, axis=1)
    if ratio:
        s_asi /= np.nanmean(r_close, axis=1)
    return pd.Series(s_asi, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_ASI48(df, **kwargs):
    return _jt_asi(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_ASI240(df, **kwargs):
    return _jt_asi(df, window=240, **kwargs)


def _jt_ddi(df, window=48, high='high', low='low', date_end=True, ratio=True):

    s_idx = df.index.values
    s_high = df[high].values
    s_low = df[low].values
    s_prev_high = df[high].shift(1).values
    s_prev_low = df[low].shift(1).values

    s_hpl = s_high + s_low
    s_prev_hpl = s_prev_high + s_prev_low
    s_max_hh_ll = np.maximum(np.abs(s_high - s_prev_high), np.abs(s_low - s_prev_low))

    s_dmz = np.where(s_hpl > s_prev_hpl, s_max_hh_ll, 0)
    s_dmf = np.where(s_hpl < s_prev_hpl, s_max_hh_ll, 0)

    r_idx = roll_window(s_idx, window=window)
    r_dmz = roll_window(s_dmz, window=window)
    r_dmf = roll_window(s_dmf, window=window)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_dmz = r_dmz[valid_rows]
        r_dmf = r_dmf[valid_rows]

    s_idx = r_idx[:, -1]
    s_dmz = np.nansum(r_dmz, axis=1)
    s_dmf = np.nansum(r_dmf, axis=1)

    s_diz = s_dmz / (s_dmz + s_dmf)
    s_dif = s_dmf / (s_dmz + s_dmf)
    s_ddi = s_diz - s_dif
    return pd.Series(s_ddi, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_DDI48(df, **kwargs):
    return _jt_ddi(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_DDI240(df, **kwargs):
    return _jt_ddi(df, window=240, **kwargs)


def _jt_chvolatility(df, window=48, high='high', low='low', date_end=True):
    """佳庆离散指标(Chaikin Volatility , 简称CVLT , VCI , CV)'又称“佳庆变异率指数” ，
    是通过测量一段时间内价格幅度平均值的变化来反映价格的离散程度。

    计算方法：
    HLEMA = EMA(highest - lowest, 10)
    ChaikinVolatility = 100 * (HLEMA (t) - HLEMA (t-10)) / HLEMA (t-10)

    - 注意：这里直接使用pandas的函数更快（但整体还是比较慢）
    """
    s_high = df[high]
    s_low = df[low]
    # FIXME: 如何跳过不需要计算的rolling行
    end_time = pd.Timestamp("15:00").time()

    def _roll_ewm(s):
        if s.index[-1].time() == end_time:
            return s.ewm(span=1).mean().values[-1]
        else:
            return np.NaN
    s_hml_ema = (s_high - s_low).rolling(window, min_periods=2).apply(_roll_ewm)
    s_hml_ema.dropna(inplace=True)
    s_chvola = s_hml_ema.replace(0, 1e-5).pct_change(int(window / 48))
    return s_chvola


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_CHVOL48(df, **kwargs):
    return _jt_chvolatility(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_CHVOL240(df, **kwargs):
    return _jt_chvolatility(df, window=240, **kwargs)


def _jt_bear_bull_power(df, ftype, window=48, close='close', high='high', low='low', date_end=True):

    s_idx = df.index.values
    s_close = df[close].values
    s_high = df[high].values
    s_low = df[low].values

    r_idx = roll_window(s_idx, window=window)
    r_close = roll_window(s_close, window=window)
    r_high = roll_window(s_high, window=window)
    r_low = roll_window(s_low, window=window)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_close = r_close[valid_rows]
        r_high = r_high[valid_rows]
        r_low = r_low[valid_rows]

    s_idx = r_idx[:, -1]
    s_high = r_high[:, -1]
    s_low = r_low[:, -1]
    s_close = r_close[:, -1]
    if ftype == 'bear':
        s_bpower = s_high - np.nanmean(r_close, axis=1)
    else:
        s_bpower = s_low - np.nanmean(r_close, axis=1)
    s_bpower /= s_close
    return pd.Series(s_bpower, index=s_idx)


def _jt_elder(df, window=48, close='close', **kwargs):
    s_bearpower = _jt_bear_bull_power(df, 'bear', window=window, close=close, **kwargs)
    s_bullpower = _jt_bear_bull_power(df, 'bull', window=window, close=close, **kwargs)
    # s_close = roll_window(df[close].values, window=window)[:, -1]
    # 20240412：收盘价已在各自子函数中除掉了
    s_elder_ray_idx = s_bearpower - s_bullpower
    return s_elder_ray_idx


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 1)
def jt_ELDER48(df, **kwargs):
    return _jt_elder(df, window=48, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 5)
def jt_ELDER240(df, **kwargs):
    return _jt_elder(df, window=240, **kwargs)


def _jt_rvi(df, window1=48, window2=10, close='close', date_end=True):

    s_idx = df.index.values
    s_close = df[close].values
    # s_prev_close = df[close].shift(1).values
    s_close_diff = df[close].diff(1).values

    r_idx = roll_window(s_idx, window=window1)
    r_close = roll_window(s_close, window=window1)
    r_close_diff = roll_window(s_close_diff, window=window1)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_close = r_close[valid_rows]
        r_close_diff = r_close_diff[valid_rows]

    s_idx = r_idx[:, -1]
    s_std = np.nanstd(r_close, axis=1)
    s_close_diff = r_close_diff[:, -1]
    s_upstd = np.where(s_close_diff > 0, s_std, 0)
    s_downstd = np.where(s_close_diff < 0, s_std, 0)
    s_up_rvi = np.nanmean(roll_window(s_upstd, window=window2), axis=1)
    s_down_rvi = np.nanmean(roll_window(s_downstd, window=window2), axis=1)
    s_idx = roll_window(s_idx, window=window2)[:, -1]
    s_rvi = s_up_rvi / (s_up_rvi + s_down_rvi + 1e-5)
    return pd.Series(s_rvi, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 20)
def jt_RVI_1_20(df, **kwargs):
    return _jt_rvi(df, window1=48, window2=20, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 25)
def jt_RVI_5_20(df, **kwargs):
    return _jt_rvi(df, window1=240, window2=20, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 40)
def jt_RVI_20_20(df, **kwargs):
    return _jt_rvi(df, window1=960, window2=20, **kwargs)


def _jt_mdd(df, window=60, close='close', excess=True):
    """区间最大回撤
    
    - 此处的最大回撤概念就是ep.max_drawdown的概念，也就是先算给定区间的current_drawdown，然后再相同区间内取最小值为该区间的max_drawdown
    """
    s_idx = df.index.values
    s_ret = df[close].pct_change(1).values
    s_bm_ret = df[f'{close}___roc__window_1'].values

    r_idx = roll_window(s_idx, window=window)
    r_naive = roll_window(s_ret, window=window)
    r_benchmark = roll_window(s_bm_ret, window)
    if excess:
        r_ex = r_naive - r_benchmark
    else:
        r_ex = r_naive
    r_cum_ex = np.cumprod(r_ex + 1, axis=1)
    r_cum_max_ex = np.maximum.accumulate(r_cum_ex, axis=1)
    r_ldd_ex_ret = (r_cum_ex - r_cum_max_ex) / r_cum_max_ex
    r_ldd_ex_ret = np.where(r_ldd_ex_ret < 0, r_ldd_ex_ret, 0)
    s_mdd_ex_ret = np.nanmin(r_ldd_ex_ret, axis=1)
    s_idx = r_idx[:,-1]
    return pd.Series(s_mdd_ex_ret, index=s_idx)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 3)
def jt_MDD16(df):
    return _jt_mdd(df, window=16, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 4)
def jt_MDD24(df):
    return _jt_mdd(df, window=24, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 6)
def jt_MDD40(df):
    return _jt_mdd(df, window=40, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 3)
def jt_MDD32(df):
    return _jt_mdd(df, window=32, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 4)
def jt_MDD48(df):
    return _jt_mdd(df, window=48, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 6)
def jt_MDD80(df):
    return _jt_mdd(df, window=80, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 3)
def jt_MDD96(df):
    return _jt_mdd(df, window=96, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 4)
def jt_MDD144(df):
    return _jt_mdd(df, window=144, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 6)
def jt_MDD240(df):
    return _jt_mdd(df, window=240, excess=True)


def _jt_sharpe_ratio(df, window=60, close='close', excess=True, date_end=False):
    s_idx = df.index.values
    s_ret = df[close].pct_change().values
    s_bm_ret = df[f'{close}___roc__window_1'].values
    s_ex_ret = s_ret - s_bm_ret

    r_idx = roll_window(s_idx, window=window)
    if excess:
        r_ex_ret = roll_window(s_ex_ret, window=window)
    else:
        r_ex_ret = roll_window(s_ret, window=window)

    if date_end:
        valid_rows = _filter_matrix_rows_by_date_ends(r_idx)
        r_idx = r_idx[valid_rows]
        r_ex_ret = r_ex_ret[valid_rows]
    
    s_idx = r_idx[:, -1]
    s_sma_ex_ret = np.nanmean(r_ex_ret, axis=1)
    s_std_ex_ret = np.nanstd(r_ex_ret, axis=1)
    s_ex_sharpe_ratio = s_sma_ex_ret / s_std_ex_ret
    return pd.Series(s_ex_sharpe_ratio, s_idx)


# @set_property('by_date', False)
# @set_property('base_frequency', "1min")
# @set_property('window', 40)
# def jt_SHARPE_RATIO(df, window=80, **kwargs):
#     return _jt_sharpe_ratio(df, window=window, **kwargs)

# @set_property('by_date', False)
# @set_property('base_frequency', "1min")
# @set_property('window', 240)
# def jt_SHARPE_RATIO240(df, **kwargs):
#     return _jt_sharpe_ratio(df, window=240, **kwargs)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 3)
def jt_SHARPE_RATIO16(df):
    return _jt_sharpe_ratio(df, window=16, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 4)
def jt_SHARPE_RATIO24(df):
    return _jt_sharpe_ratio(df, window=24, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 6)
def jt_SHARPE_RATIO40(df):
    return _jt_sharpe_ratio(df, window=40, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 3)
def jt_SHARPE_RATIO32(df):
    return _jt_sharpe_ratio(df, window=32, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 4)
def jt_SHARPE_RATIO48(df):
    return _jt_sharpe_ratio(df, window=48, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 6)
def jt_SHARPE_RATIO80(df):
    return _jt_sharpe_ratio(df, window=80, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 3)
def jt_SHARPE_RATIO96(df):
    return _jt_sharpe_ratio(df, window=96, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 4)
def jt_SHARPE_RATIO144(df):
    return _jt_sharpe_ratio(df, window=144, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 6)
def jt_SHARPE_RATIO240(df):
    return _jt_sharpe_ratio(df, window=240, excess=True)



@set_property('by_date', False)
def _jt_roc(df, close='close', n=1, excess=True):
    """简单收益率（含超额选项）"""
    _naive_ret = df[close].pct_change(n)
    if not excess:
        return _naive_ret
    else:
        _bm_ret = df[f'{close}___roc__window_{n}']
        return _naive_ret - _bm_ret


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 3)
def jt_ROC16(df):
    return _jt_roc(df, n=16, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 4)
def jt_ROC24(df):
    return _jt_roc(df, n=24, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "30min")
@set_property('window', 6)
def jt_ROC40(df):
    return _jt_roc(df, n=40, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 3)
def jt_ROC32(df):
    return _jt_roc(df, n=32, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 4)
def jt_ROC48(df):
    return _jt_roc(df, n=48, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "15min")
@set_property('window', 6)
def jt_ROC80(df):
    return _jt_roc(df, n=80, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 3)
def jt_ROC96(df):
    return _jt_roc(df, n=96, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 4)
def jt_ROC144(df):
    return _jt_roc(df, n=144, excess=True)


@set_property('by_date', False)
@set_property('base_frequency', "5min")
@set_property('window', 6)
def jt_ROC240(df):
    return _jt_roc(df, n=240, excess=True)
