import re
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from dataclasses import dataclass
from loguru import logger
from retrying import retry
from deprecation import deprecated
import rqdatac
# rqdatac.init(13601611030,'PB123456789')
# rqdatac.init(username='18101949790', password='123456')
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)

from rqfactor import *
from rqfactor.extension import (
    rolling_window, RollingWindowFactor, CombinedRollingWindowFactor, CombinedFactor,
    UserDefinedLeafFactor, UnaryCrossSectionalFactor, CombinedCrossSectionalFactor
)
from numpy.linalg import LinAlgError
from scipy.ndimage import shift
from scipy import stats
import math
import tsfresh.feature_extraction.feature_calculators as tsfresh_calculators

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
import warnings
warnings.filterwarnings('ignore')

DEBUG = True
KEEPNAN = True
    
ALL_FACTOR_NAMES = rqdatac.get_all_factor_names()
# 三大表基础因子
ALL_BS_FACTOR_NAMES = rqdatac.get_all_factor_names(type='balance_sheet')
ALL_CFS_FACTOR_NAMES = rqdatac.get_all_factor_names(type='cash_flow_statement')
ALL_IS_FACTOR_NAMES = rqdatac.get_all_factor_names(type='income_statement')
ALL_BASIC_FIN_FACTOR_NAMES = ALL_BS_FACTOR_NAMES + ALL_CFS_FACTOR_NAMES + ALL_IS_FACTOR_NAMES
# 财务衍生因子
ALL_EOD_FACTOR_NAMES = rqdatac.get_all_factor_names(type="eod_indicator")  # 估值指标
ALL_OPR_FACTOR_NAMES = rqdatac.get_all_factor_names(type="operational_indicator")  # 经营衍生指标
ALL_CFI_FACTOR_NAMES = rqdatac.get_all_factor_names(type="cash_flow_indicator")  # 现金流衍生指标
ALL_FIN_FACTOR_NAMES = rqdatac.get_all_factor_names(type="financial_indicator")  # 财务衍生指标
ALL_GRO_FACTOR_NAMES = rqdatac.get_all_factor_names(type="growth_indicator")  # 增长衍生指标
# 自定义财务衍生因子
ALL_SELF_DEFINED_FACTOR_NAMES = [
    'inventory_turnover_days_ttm',
    'long_loans_to_asset_ttm',
    'financial_expense_rate_ttm',
    'admin_expense_rate_ttm',
    'net_ocf_to_operating_net_income_ttm',
    'assi_ttm',
    'bonds_payable_to_asset_ttm',
    'cash_of_sales_ttm',
    'long_debt_to_asset_ttm',
    'operating_profit_ratio_ttm',
    'net_ocf_to_current_liabilities_ttm',
    'sales_service_cash_to_or_ttm',
    'tax_ratio_ttm',
    'total_profit_cost_ratio_ttm',
    'cfo_to_ev_ttm',
    'acca_ttm',
]
ALL_DERI_FIN_FACTOR_NAMES = ALL_EOD_FACTOR_NAMES + ALL_OPR_FACTOR_NAMES + ALL_CFI_FACTOR_NAMES + ALL_FIN_FACTOR_NAMES + ALL_GRO_FACTOR_NAMES + ALL_SELF_DEFINED_FACTOR_NAMES


def _rolling_median(series, window):
    return np.nanmedian(rolling_window(series, window), axis=1)


def MEDIAN(factor, window):
    return RollingWindowFactor(_rolling_median, window, factor)


class MyRQFactorsBase(object):

    def __init__(self):
        self.name = 'rqfactors'
        self.factor_list = self.get_factor_list()
        self._rq_all_factors = rqdatac.get_all_factor_names()

    def __getitem__(self, key: str, label=False, **kwargs):
        if key in self._rq_all_factors:
            factor = Factor(key)
        else:
            if not label:
                key = 'rq_' + key
            factor = self.__getattribute__(key)(**kwargs)
        return factor

    def get_factor_list(self, labeled_funcs=False, additional=True):
        all_funcs = dir(self)
        factor_funcs = [func for func in all_funcs if func.startswith('rq_')]
        if labeled_funcs:
            return factor_funcs
        else:
            no_label_funcs = [func.split('_', 1)[-1] for func in factor_funcs]
            return no_label_funcs

    def _check_factor_exists(self, factor_name):
        if factor_name not in self._rq_all_factors:
            if factor_name not in self.factor_list:
                raise ValueError("暂不支持因子 %s，请重新输入！" % factor_name)

    def get_factor(self, factor_name, funcs=(), ts_kwargs={}, **kwargs):
        if factor_name in self._rq_all_factors:
            return Factor(factor_name)
        has_ts_func = False
        # 字符串内搜索是否包含时序算法
        _methods = self.get_adjust_methods()
        for method in _methods:
            if method in factor_name:
                # FIXME: 存在method的重复，例如 has_duplicate 和 has_duplicate_max，当前的识别方法存在错误识别的风险
                base_factor_name, window = factor_name.split(method)
                # self._check_factor_exists(base_factor_name)
                logger.warning("即将获取基础因子<{}>的<{}>版本，窗口为: {}".format(base_factor_name, method, window))
                factor = self.__getattribute__(method)(self.__getitem__(base_factor_name, **kwargs), window)
                logger.warning("即将返回基础因子<{}>的<{}>版本，窗口为: {}".format(base_factor_name, method, window))
                # factor_name += method
                has_ts_func = True
                # return factor
        
        if not has_ts_func:
            # 若输入的因子名不含时序操作,则先判断后搜索返回因子即可
            self._check_factor_exists(factor_name)
            factor = self.__getitem__(factor_name, **kwargs)

        # 判断是否传入list
        for func in funcs:
            if 'c3' in func:
                matches = re.findall('(c3)(\d+)', func)
            else:
                matches = re.findall('([a-zA-Z_]+)(\d+)', func)
            if len(matches) == 0:
                raise ValueError(f"非法时序算法:{func}")
            method, window = matches[0]
            if '_' + method in self.get_tsf_adjust_methods():
                factor = self.__getattribute__('_' + method)(factor, window, **ts_kwargs)
                logger.warning("对因子<{}>执行TSFresh<{}>运算，窗口为: {}，参数为{}".format(factor_name, method, window, ts_kwargs))
            else:
                factor = self.__getattribute__('_' + method)(factor, window)
                logger.warning("对因子<{}>执行<{}>运算，窗口为: {}".format(factor_name, method, window))
            factor_name += '_' + func
        print(funcs, factor_name, factor)
        return factor

    @staticmethod
    def _signed_series(series: Series or DataFrame, initial: int = None) -> Series or DataFrame:
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

    def get_adjust_methods(self):
        """时序调整函数

        """
        _methods = [
            '_demean', '_osc',
            '_divmean',
            '_zscore',
            '_delta', '_revs',
            '_roc','_froc',
            '_ma', '_ema', '_dma',
            '_slope', '_ldd',
            '_mdd',
            '_pctchg', '_volt',
        ]

        _ts_methods = self.get_tsf_adjust_methods()
        return _methods + _ts_methods

    @staticmethod
    def get_tsf_adjust_methods():
        _ts_methods = []

        for func_name in dir(tsfresh_calculators):
            if func_name.startswith('_') or func_name == 'set_property':
                continue
            _ts_methods.append(f'_{func_name}')
        return _ts_methods

    @staticmethod
    def _demean(factor: Factor, window=20):
        """去均值"""
        if not isinstance(window, int):
            window = int(window)
        return factor - MA(factor, window)

    def _osc(self, factor, window=20):
        """变动速率线

        - 本质定义就是当前值减去过去N日的均值，与_demean含义相同
        """
        return self._demean(factor, window)

    @staticmethod
    def _divmean(factor: Factor, window=20):
        if not isinstance(window, int):
            window = int(window)
        return factor / MA(factor, window)

    @staticmethod
    def _delta(factor, window=1):
        if not isinstance(window, int):
            window = int(window)
        return DELTA(factor, window)

    @staticmethod
    def _zscore(factor, window=240):
        if not isinstance(window, int):
            window = int(window)
        return TS_ZSCORE(factor, window)

    @staticmethod
    def _abs(factor):
        return ABS(factor)

    @staticmethod
    def _revs(factor, window=1):
        """N日动量
        计算方法：REVS=Close/delay(close, N)

        - 20231007：定义同ROC有细微不同，不是一个pct概念，单纯是个比值概念
        """
        if not isinstance(window, int):
            window = int(window)
        return factor / REF(factor, window)

    @staticmethod
    def _roc(factor, window=1):
        """N日变动比率（百分比）
        计算方法：ROC=100 * delta(factor, N)/ref(factor, N)
        """
        if not isinstance(window, int):
            window = int(window)
        return 100 * DELTA(factor, window) / REF(factor, window)

    @staticmethod
    def _froc(factor, window=1):
        """N日变动比率（百分比）
        计算方法：ROC=100 * delta(factor, N)/ref(factor, N)
        """

        if not isinstance(window, int):
            window = int(window)
        return 100 * DELTA(factor, window) / REF(factor, -window)

    @staticmethod
    def _pctchg(factor, window=1):

        if not isinstance(window, int):
            window = int(window)
        return PCT_CHANGE(factor, window)

    @staticmethod
    def _ma(factor, window=1):
        if not isinstance(window, int):
            window = int(window)
        return MA(factor, window)

    @staticmethod
    def _ema(factor, window=1):
        if not isinstance(window, int):
            window = int(window)
        return EMA(factor, window)

    @staticmethod
    def _dma(factor, window=1):
        if not isinstance(window, int):
            window = int(window)
        return DMA(factor, window, 1 / window)

    @staticmethod
    def _slope(factor, window=2):
        """最近 n 期因子值的斜率，与 talib 中 LINEARREG_SLOPE 相同"""
        if not isinstance(window, int):
            window = int(window)
        return SLOPE(factor, window)

    @staticmethod
    def _volt(factor, window=2):
        if not isinstance(window, int):
            window = int(window)
        return STD(factor, window)

    @staticmethod
    def _mdd(factor, window=2):
        if not isinstance(window, int):
            window = int(window)

        _cummax = TS_MAX(factor, window)
        _cummin = TS_MIN(factor, window)
        return ABS(_cummin / _cummax - 1)

    @staticmethod
    def _ldd(factor, window=2):
        if not isinstance(window, int):
            window = int(window)

        _cummax = TS_MAX(factor, window)
        return ABS(factor / _cummax - 1)

    # @staticmethod
    # def _abs_to_quantile(factor, window=20):
    #     if not isinstance(window, int):
    #         window = int(window)
    #     _quantile = QUANTILE()
    #     return

    @staticmethod
    def _get_factor_zeros(order_book_ids, start_date, end_date):
        """0值-因子构建"""
        trading_dates = rqdatac.get_trading_dates(start_date, end_date)
        result = pd.DataFrame(np.zeros((len(trading_dates), len(order_book_ids))),
                              index=pd.to_datetime(trading_dates),
                              columns=order_book_ids)
        return result

    def _rq_zeros(self):
        """0值矩阵只作为因子中间变量计算使用，不提供外部调用"""
        return UserDefinedLeafFactor('zeros', self._get_factor_zeros)

    @staticmethod
    def _get_factor_yield_curve(order_book_ids, start_date, end_date):
        """10年期国债收益率-因子构建

        - 国债收益率的日期和交易日并非一一对应，这里要做一个筛选
        """
        trading_dates = rqdatac.get_trading_dates(start_date, end_date)
        no_risk_ret = rqdatac.get_yield_curve(start_date, end_date, tenor='10Y')
        no_risk_ret = no_risk_ret[no_risk_ret.index.isin(trading_dates)]
        datas = [no_risk_ret] * len(order_book_ids)
        result = pd.concat(datas, ignore_index=True, axis=1)
        result.index = no_risk_ret.index
        result.columns = order_book_ids
        return result

    def rq_yield_curve(self):
        return UserDefinedLeafFactor('yield_curve', self._get_factor_yield_curve)

    def rq_ex_sharpe_ratio(self, window=2):
        """夏普比率计算"""
        if window == 1:
            return self._rq_zeros()
        _ex_ret = self.rq_excess_zz1000_return()
        _sma = self.ma(_ex_ret, window=window)
        _std = STD(_ex_ret, window=window)
        return _sma / _std

    def rq_excess_zz1000_return(self):
        """个股相对于中证1000（000852.XSHG））的超额收益率"""
        zz1000_return = self.rq_zz1000_return()
        naive_return = self.rq_naive_return()
        return naive_return - zz1000_return

    def rq_zz1000_return(self):
        """中证1000（000852.XSHG）简单收益率-因子调用
        """
        return self._rq_benchmark_return('000852.XSHG')
    def rq_naive_return(self, base='close', adjust='post'):
        """个股简单日度收益率

        :param base: 计算基准，默认前一日收盘价，未来可支持当日开盘价
        :param adjust: 除权/复权操作，默认后复权，未来可支持前复权（前复权的因子名与close不同）
        """
        return Factor('close') / REF(Factor('close'), 1) - 1

    def rq_excess_return(self):
        """个股日度超额收益率（基于无风险收益率）
        - 由于是两个自建因子的复合构建，存在QuotaExceedError的可能性，改进方案是写进一个get_xxx函数返回减去无风险收益率的超额收益率
        """
        # risk_free_return = self.rq_yield_curve()
        naive_return = self.rq_naive_return()
        # return naive_return - risk_free_return
        return naive_return

    def _rq_benchmark_return(self, benchmark_order_book_id):
        close_ = FIX(Factor('close'), benchmark_order_book_id)
        return close_ / REF(close_, 1) - 1

    def rq_zz500_return(self):
        """中证500（000905.XSHG）简单收益率-因子调用
        - FIXMEd: 改为米筐自带的FIX算子返回，大致思路如下
           FIX(Factor('close'), '000905.XSHG')/REF(FIX(Factor('close'), '000905.XSHG')) - 1
           可以测试两种方法读取的数据是否一致 -> 已验证完全一致
        """
        return self._rq_benchmark_return('000905.XSHG')

    def rq_excess_zz500_return(self):
        """个股超额收益率（基于中证500（000905.XSHG））
        - 基于自定义zz500_return和rq_naive_return因子计算
        - 由于是两个自建因子的复合构建，存在QuotaExceedError的可能性
        """
        zz500_return = self.rq_zz500_return()
        naive_return = self.rq_naive_return()
        return naive_return - zz500_return

    def rq_zz500_excess_return(self, ):
        """中证500相对于无风险收益率的超额收益率-因子调用"""
        zz500_return = self.rq_zz500_return()
        no_risk_return = self.rq_yield_curve()
        return zz500_return - no_risk_return

    @staticmethod
    def _get_zz500_pe_ttm(order_book_ids, start_date, end_date):
        trading_dates = rqdatac.get_trading_dates(start_date, end_date)
        index_pe_ttm = rqdatac.index_indicator('000905.XSHG', start_date, end_date, 'pe_ttm').reset_index(0, drop=True)
        index_pe_ttm = index_pe_ttm[index_pe_ttm.index.isin(trading_dates)]
        datas = [index_pe_ttm] * len(order_book_ids)
        result = pd.concat(datas, ignore_index=True, axis=1)
        result.index = index_pe_ttm.index
        result.columns = order_book_ids
        return result

    def rq_zz500_pettm(self):
        """获取中证500（000905.XSHG）的PE(TTM)数据"""
        return UserDefinedLeafFactor('zz500_pe_ttm', self._get_zz500_pe_ttm)

    def rq_total_turnover(self):
        """成交额

        - 实际米筐支持Factor获取，但是不在rqdatac.get_all_factor_names()中，所以特此单列
        - 后续的close等基本行情因子同理
        """
        return Factor('total_turnover')

    def rq_volume(self):
        """成交量"""
        return Factor("volume")

    def rq_close(self):
        """收盘价（后复权）"""
        return Factor('close')

    def rq_open(self):
        """开盘价（后复权）"""
        return Factor('open')

    def rq_high(self):
        """最高价（后复权）"""
        return Factor('high')

    def rq_low(self):
        """最低价（后复权）"""
        return Factor('low')

    def rq_close_roc(self, window = 1):
        """收盘价（后复权）"""
        _close = Factor('close')
        _close_roc = self._roc(_close, window)
        return _close_roc

    def rq_turnover(self):
        """换手率因子-基于成交额计算

        计算公式: 成交额/流通市值
        - market_cap_2: 流通市值 = close * CAPITAL
        """
        return Factor('total_turnover') / Factor('market_cap_2')

    def rq_hsl(self):
        """换手率因子-基于成交量计算

        计算公式: 成交量/流通股本
        """
        return Factor('volume') / Factor('capital') * 100

    @staticmethod
    def _get_circulation_a_shares(order_book_ids, start_date, end_date):
        # 这里需要reindex，否则返回的数据列数未对齐会报错
        df = rqdatac.get_shares(order_book_ids, start_date, end_date, 'circulation_a', expect_df=False).reindex(columns=order_book_ids)
        return df

    def rq_circulating_shares(self):
        return UserDefinedLeafFactor('circulating_shares', self._get_circulation_a_shares)

    def rq_udl(self, window1=3, window2=5, window3=10, window4=20):
        """引力线

        计算公式：
        UDL = (MA(CLOSE,N1)+MA(CLOSE,N2)+MA(CLOSE,N3)+MA(CLOSE,N4))/4
        """
        close = Factor('close')
        return (MA(close, window1) + MA(close, window2) +
                MA(close, window3) + MA(close, window4)) / 4
    @staticmethod
    def ma(factor, window=1):
        if not isinstance(window, int):
            window = int(window)
        return MA(factor, window)

    def rq_sdk(self):
        """随即慢速指标

        计算公式
        LOWV = LLV(LOW, N)
        HIGHV = HHV(HIGH, N)
        RSV = EMA((CLOSE – LOWV) / (HIGHV – LOWV) * 100, M)
        SKD_K = EMA(RSV , M)
        SKD_D = MA(SKD_K, M)
        """


def create_ts_method(ts_function):
    def _rolling_ts_method(series, window):
        # 从tsfresh文件中查找对应函数
        ts_func = getattr(tsfresh_calculators, ts_function)
        # TSFresh的函数分为两类：simple和combiner
        # simple类指作用于series，输入简单参数，直接返回数值
        # combiner类指作用于series，输入多组复合参数，返回多组zip的结果，每个结果返回形式为(param_name, value)
        if ts_func.fctype == 'simple':
            def ts_func_simple(x, **kwargs):
                # 避免全是NaN值报错，如果全NaN则直接返回NaN
                if np.all(np.isnan(x)):
                    return np.NaN
                result = ts_func(x, **kwargs)
                return result
            if len(ts_kwargs) > 0:
                return np.apply_along_axis(ts_func_simple, 1, rolling_window(series, window), **ts_kwargs)
            else:
                # 不需要参数的直接apply
                return np.apply_along_axis(ts_func_simple, 1, rolling_window(series, window))
        elif ts_func.fctype == 'combiner':
            def ts_func_combiner(x, **kwargs):
                # 避免全是NaN值报错，如果全NaN则直接返回NaN
                if np.all(np.isnan(x)):
                    return np.NaN
                result = ts_func(x, [kwargs])
                return list(result)[0][1]
            return np.apply_along_axis(ts_func_combiner, 1, rolling_window(series, window), **ts_kwargs)

    def ts_method(self, factor, window, **kwargs):
        window = int(window)
        global ts_kwargs
        ts_kwargs = kwargs
        return RollingWindowFactor(_rolling_ts_method, window, factor)
    return ts_method


for func in dir(tsfresh_calculators):
    if func.startswith('_') or func == 'set_property':
        continue
    method_name = f'_{func}'
    method = create_ts_method(func)
    setattr(MyRQFactorsBase, method_name, method)


@dataclass
class FinancialFactorName:
    factor_name: str  # 因子名
    state_code: str  # 状态码，包括_ttm, _mrq, _lyr
    periods: int  # 期数，默认第0期，目前不会在期数上加减，所以直接保留字符串格式
    ftype: str  # 因子类型，默认None值，目前支持base, deri的输入。

    @classmethod
    def from_string(cls, string: str, ftype=None):
        for s in ['_yoy', '_mom', '_byoy', '_bmom']:
            if s in string:
                f, sc, p, _ = string.rsplit('_', 3)
                break
        else:
            f, sc, p = string.rsplit('_', 2)
        if sc == 'lf':
            sc = 'mrq'
        #print(f,sc,p,string)
        return cls(factor_name=f, state_code=sc, periods=p, ftype=ftype)

    def __post_init__(self):
        self.periods = int(self.periods)
        if self.state_code not in ['ttm', 'mrq', 'lyr', 'lf']:
            raise ValueError("目前只支持_ttm, _mrq, _lyr, _lf的状态码，请确认输入状态码无误。")
        if self.ftype is None:
            logger.debug("未输入因子类别，将基于因子名称推断所属类别")
            deri_factor_name = self.factor_name + '_' + self.state_code
            if deri_factor_name in ALL_DERI_FIN_FACTOR_NAMES:
                self.ftype = 'deri'
            # 这里因为米筐的三大表接口返回的因子列表不全，所以基础因子在衍生财务因子之后判断，且判断用总因子表判断
            # 逻辑为：如果不是衍生财务因子，再看是否属于支持的因子，如果还不属于，那就肯定都不支持
            elif self.factor_name in ALL_FACTOR_NAMES:
                self.ftype = 'base'
            else:
                raise ValueError(f"不支持因子{self.factor_name}")
        self.base_factor = self.to_string(0)

    def shift(self, periods):
        """调整因子周期

        - 20231122：目前实际没有频繁使用，因为如果涉及到periods的迁移，直接to_string中设置对应期数即可
        """
        self.periods += periods
        return self

    def yoy_formula(self):
        last_factor = self.to_string(self.periods + 4)
        factor = self.to_string()
        yoy_formula = f'({factor} - {last_factor})/abs({last_factor})'
        return yoy_formula

    def mom_formula(self):
        last_factor = self.to_string(self.periods + 1)
        factor = self.to_string()
        return f'({factor} - {last_factor})/abs({last_factor})'

    def byoy_formula(self):
        factor = self.to_string()
        last_factor = self.to_string(self.periods + 1)
        sum_factor = f'{factor} + {last_factor}'

        last4_factor = self.to_string(self.periods + 4)
        last5_factor = self.to_string(self.periods + 5)
        sum_last4_factor = f'{last4_factor} + {last5_factor}'
        return f'({sum_factor} - ({sum_last4_factor}))/abs({sum_last4_factor})'

    def bmom_formula(self):
        factor = self.to_string()
        last_factor = self.to_string(self.periods + 1)
        sum_factor = f'{factor} + {last_factor}'

        last2_factor = self.to_string(self.periods + 2)
        last3_factor = self.to_string(self.periods + 3)
        sum_last2_factor = f'{last2_factor} + {last3_factor}'
        return f'({sum_factor} - ({sum_last2_factor}))/abs({sum_last2_factor})'

    def to_string(self, period=None, sep='_'):
        if period is None:
            period = self.periods
        return sep.join([self.factor_name, self.state_code, str(period)])


class Fundamentals(MyRQFactorsBase):

    def __init__(self):
        super().__init__()
        self.dict_fff = self.get_financial_factors_formulas()

    @staticmethod
    def get_financial_factors_formulas():
        # fp_formulas = r"\\192.168.1.168\samba\Market\rqfactors\米筐_衍生财务指标_定制化指标_穿透版_含中文释义_20231122.xlsx"
        fp_formulas = r"\\192.168.3.100\samba\Market\rqfactors\米筐_衍生财务指标_定制化指标_穿透版_含中文释义_20231122.xlsx"
        all_sheets_dict = pd.read_excel(fp_formulas, sheet_name=None)
        # 删除估值有关指标-估值有关指标目前不支持向前回溯
        # all_sheets_dict.pop('估值有关指标')  # 穿透版不需要删除
        # 将所有工作表合并为一个DataFrame
        all_data = pd.concat(all_sheets_dict, ignore_index=True)[['字段', '公式']]
        all_data['字段'] = all_data['字段'].str.replace('_lf', '_mrq') + '_0'  # 无日期后缀的都添加初始日期后缀_0表明可以支持前推
        return all_data.set_index('字段')['公式'].to_dict()

    @staticmethod
    def _to_rq_expression(formula):
        """给定公式返回因子

        - 20231025：目前只支持简单公式处理，包括A/B等形式
        - 20231122：已弃用，后续均使用_to_rq_expression_plus处理
        """
        pat_components = re.compile('[a-zA-Z0-9_]+')
        _computes = ['log', 'abs']
        matches = pat_components.findall(formula)
        eval_expr = formula
        for match in matches:
            if match in _computes:
                replacement = match.upper()
            elif match.isnumeric():
                continue
            else:
                replacement = "Factor('" + match + "')"
            eval_expr = eval_expr.replace(match, replacement)
        if DEBUG:
            logger.debug(eval_expr)
        return eval(eval_expr)

    @staticmethod
    def _to_rq_expression_plus(formula, periods=0):
        """给定公式返回因子（支持多周期回溯）

        """
        pat_components = re.compile('[a-zA-Z0-9_]+')
        _computes = ['log', 'abs']
        matches = pat_components.findall(formula)
        # v2: 分别替换后合并
        pat_split = re.compile('(' + '|'.join(matches) + ')')
        splits = pat_split.split(formula)
        replaced = []
        for split in splits:
            if split in matches:
                match = split
                if match in _computes:
                    replacement = match.upper()
                elif match.isnumeric():
                    replacement = match
                else:
                    ff_match = FinancialFactorName.from_string(match)
                    new_match = ff_match.to_string(ff_match.periods + periods)
                    replacement = "Factor('" + new_match + "')"
            else:
                replacement = split
            replaced.append(replacement)

        eval_expr = ''.join(replaced)
        if DEBUG:
            logger.debug(eval_expr)
        return eval_expr

    def _yoy(self, factor_name):
        """因子同比计算"""
        ff = FinancialFactorName.from_string(factor_name)
        ff_base = ff.base_factor
        base_formula = self.dict_fff.get(ff_base, None)
        if not base_formula:
            # 这里base_ff的yoy已经做了shift，所以_to_rq_expressions内不需要再shift
            return self._to_rq_expression_plus(ff.yoy_formula())
        else:
            f0 = self._to_rq_expression_plus(base_formula, periods=ff.periods)
            f4 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 4)
            return f'(({f0}) - ({f4}))/abs(({f4}))'

    def _mom(self, factor_name):
        """因子环比计算"""
        ff = FinancialFactorName.from_string(factor_name)
        ff_base = ff.base_factor
        base_formula = self.dict_fff.get(ff_base, None)
        if not base_formula:
            return self._to_rq_expression_plus(ff.mom_formula())
        else:
            f0 = self._to_rq_expression_plus(base_formula, periods=ff.periods)
            f1 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 1)
            return f'(({f0}) - ({f1}))/abs(({f1}))'

    def _byoy(self, factor_name):
        """因子连续两期同比计算"""
        ff = FinancialFactorName.from_string(factor_name)
        ff_base = ff.base_factor
        base_formula = self.dict_fff.get(ff_base, None)
        if not base_formula:
            return self._to_rq_expression_plus(ff.byoy_formula())
        else:
            f0 = self._to_rq_expression_plus(base_formula, periods=ff.periods)
            f1 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 1)
            sum_f01 = f'{f0} + {f1}'
            f4 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 4)
            f5 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 5)
            sum_f45 = f'{f4} + {f5}'
            return f'({sum_f01} - ({sum_f45}))/abs({sum_f45})'

    def _bmom(self, factor_name):
        """因子连续两期环比计算

        - 注意，如果是_mrq数据，则这里的连续两期环比总共是四期不交叉的mrq数据，进一步推理连续四期环比其实就是ttm的同比
               如果是_ttm数据，则连续两期环比的实际构成存在交叉，但命名上不存在，例如_ttm_0+_ttm_1和_ttm_2+_ttm_3的连续两期环比。
        """
        ff = FinancialFactorName.from_string(factor_name)
        ff_base = ff.base_factor
        base_formula = self.dict_fff.get(ff_base, None)
        if not base_formula:
            return self._to_rq_expression_plus(ff.bmom_formula())
        else:
            f0 = self._to_rq_expression_plus(base_formula, periods=ff.periods)
            f1 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 1)
            sum_f01 = f'{f0} + {f1}'
            f2 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 2)
            f3 = self._to_rq_expression_plus(base_formula, periods=ff.periods + 3)
            sum_f23 = f'{f2} + {f3}'
            return f'({sum_f01} - ({sum_f23}))/abs({sum_f23})'

    def get_financial_factor(self, factor_name):
        """针对财务因子（含多状态、多周期判断）的调用函数


        FIXMEd: 目前这样的写法只支持基础因子的yoy或者mom，衍生因子目前也支持前推，因此理论上衍生因子也需要支持yoy
         解决思路：重新定义一个内置函数_yoy，接收因子名称，判断是否是基础财务因子，基础财务因子直接调用ff.yoy_formula()，不是则查找公式，调用ff.yoy_formula后替换为具体公式
        """
        if factor_name in ALL_FACTOR_NAMES:
            return Factor(factor_name)
        # v2: 同时支持衍生因子和原生因子的尾缀功能
        # 首先判断是否存在尾缀调用，存在则转至对应函数处理
        _sub_funcs = ['_yoy', '_mom', '_byoy', '_bmom']
        for sub_func in _sub_funcs:
            if sub_func in factor_name:
                shifted_expr = eval(f'self.{sub_func}(factor_name)')
                break
        else:
            ff = FinancialFactorName.from_string(factor_name)
            ff_base = ff.base_factor
            base_formula = self.dict_fff.get(ff_base, None)
            # print(base_formula)
            shifted_expr = self._to_rq_expression_plus(base_formula, periods=ff.periods)
        return eval(shifted_expr)

    def rq_operating_expense_rate(self):
        """营业费用与营业总收入之比（Operating expense rate）。
        计算方法：营业费用与营业总收入之比=销售费用（TTM）/营业总收入（TTM）。
        """
        return self._to_rq_expression('selling_expense_ttm_0/revenue_ttm_0')

    def rq_inventory_tdays(self):
        """存货周转天数（Inventory turnover days）。
        计算方法：存货周转天数=360/存货周转率。

        - 20231007: 官网虽然有写inventory_turnover_ttm_0这个变量，但实际只有inventory_turnover_ttm这个因子
          + 20231120：米筐已更新，统一为inventory_turnover_ttm变量。
        - 20231122：已写入'米筐_衍生财务指标_20231117_定制化指标.xlsx'的自定义衍生指标中，此处调用添加尾缀_0
        """
        # return self._to_rq_expression('360/inventory_turnover_ttm')
        return self.get_financial_factor('inventory_turnover_days_ttm_0')

    def rq_lcap(self):
        """对数市值（Natural logarithm of total market values）。
        计算方法：对数市值=市值的对数
        """
        return self._to_rq_expression('log(market_cap_3+1)')

    def rq_lflo(self):
        """对数流通市值（Natural logarithm of float market values）。
        计算方法：对数流通市值=流通市值的对数。
        """
        return self._to_rq_expression('log(market_cap_2+1)')

    def rq_long_loans_to_asset(self):
        """长期借款与资产总计之比（Long term loan to total assets）。
        计算方法：长期借款与资产总计之比=长期借款/总资产

        - 20231007: long_term_loans_ttm_0 NaN值较多
        - 20231122：已写入'米筐_衍生财务指标_20231117_定制化指标.xlsx'的自定义衍生指标中，此处调用添加尾缀_0
        """
        # return self._to_rq_expression('long_term_loans_ttm_0/total_assets_ttm_0')
        return self.get_financial_factor('long_loans_to_asset_ttm_0')

    def rq_financial_expense_rate(self):
        """财务费用与营业总收入之比（Financial expense rate）。
        计算方法：财务费用与营业总收入之比=财务费用（TTM）/营业总收入（TTM

        - 20231122：已写入'米筐_衍生财务指标_20231117_定制化指标.xlsx'的自定义衍生指标中，此处调用添加尾缀_0
        """
        # return self._to_rq_expression('financing_expense_ttm_0/revenue_ttm_0')
        return self.get_financial_factor('financial_expense_rate_ttm_0')

    def rq_ctop(self):
        """现金流市值比（Cash flow to price）。
        计算方法：现金流市值比=每股派现（税前）×分红前总股本/总市值
        """
        return self._to_rq_expression('(cash_flow_from_operating_activities_ttm_0 + cash_flow_from_investing_activities_ttm_0 + cash_flow_from_financing_activities_ttm_0)/market_cap_3')

    def rq_admin_expense_rate(self):
        """管理费用与营业总收入之比

        - 20231122：已写入'米筐_衍生财务指标_20231117_定制化指标.xlsx'的自定义衍生指标中，此处调用添加尾缀_0
        """
        # return self._to_rq_expression('ga_expense_ttm_0/revenue_ttm_0')
        return self.get_financial_factor('admin_expense_rate_ttm_0')

    def rq_net_ocf_to_operating_net_income(self):
        """经营活动产生的现金流量净额与经营活动净收益之比。
        计算方法：
        经营活动产生的现金流量净额与经营活动净收益之比=经营活动产生的现金流量净额（TTM）/(营业总收入（TTM）-营业总成本（TTM）)

        - 20231122：已写入'米筐_衍生财务指标_20231117_定制化指标.xlsx'的自定义衍生指标中，此处调用添加尾缀_0
        """
        # return self._to_rq_expression('cash_flow_from_operating_activities_ttm_0/(revenue_ttm_0-total_expense_ttm_0)')
        return self.get_financial_factor('net_ocf_to_operating_net_income_ttm_0')

    @deprecated(deprecated_in="1.0", removed_in="2.0", details='使用rq_net_ocf_to_operating_net_income')
    def rq_net_ocf_to_opration_net_income(self):
        return self.rq_net_ocf_to_operating_net_income()

    def rq_assi(self):
        """对数总资产（Natural logarithm of total assets）。
        计算方法：对数总资产=总资产的对数。

        - 20231008：避免0值，求对数前+1
        """
        # return self._to_rq_expression('log(total_assets_ttm_0+1)')
        return self.get_financial_factor('assi_ttm_0')

    def rq_bonds_payable_to_asset(self):
        """应付债券与总资产之比

        - 20230926: 在标准化后目前feature_importance最高，但是采取的是空值填充为0的策略，不知道有没有影响
        - 20231009：
          + bond_payable_ttm_0 本身如果是0有公司也会披露（例如002169，2023半年报），所以这里不能填充
          + 因此如果后续使用中发现NaN值过多，考虑删除此因子
        """
        # return self._to_rq_expression('bond_payable_ttm_0/total_assets_ttm_0')
        return self.get_financial_factor('bonds_payable_to_asset_ttm_0')

    def rq_cash_of_sales(self):
        """经营活动产生的现金流量净额与营业收入之比（Cash rate of sales）。
        计算方法：经营活动产生的现金流量净额与营业收入之比=经营活动产生的现金流量净额（TTM）/营业收入（TTM）
        """
        # return self._to_rq_expression('cash_flow_from_operating_activities_ttm_0/operating_revenue_ttm_0')
        return self.get_financial_factor('cash_of_sales_ttm_0')

    def rq_long_debt_to_asset(self):
        """长期负债与资产总计之比（Long term debt to total assets）。
        计算方法：长期负债与资产总计之比=非流动负债合计/总资产。
        """
        # return self._to_rq_expression('non_current_liabilities_ttm_0/total_assets_ttm_0')
        return self.get_financial_factor('long_debt_to_asset_ttm_0')

    def rq_operating_profit_ratio(self):
        """营业利润率（Operating profit ratio），
        计算方法：营业利润率=营业利润（TTM）/营业收入（TTM）。
        """
        # return self._to_rq_expression('profit_from_operation_ttm_0/operating_revenue_ttm_0')
        return self.get_financial_factor('operating_profit_ratio_ttm_0')

    def rq_net_ocf_to_current_liabilities(self):
        """现金流动负债比（Cash provided by operations to current liability）。
        计算方法：现金流动负债比=经营活动产生的现金流量净额（TTM）/流动负债合计。
        """
        # return self._to_rq_expression('cash_flow_from_operating_activities_ttm_0/current_liabilities_ttm_0')
        return self.get_financial_factor('net_ocf_to_current_liabilities_ttm_0')

    def rq_sales_service_cash_to_or(self):
        """销售商品提供劳务收到的现金与营业收入之比（Sale service cash to operating revenues）。
        计算方法：销售商品提供劳务收到的现金与营业收入之比=销售商品和提供劳务收到的现金（TTM）/营业收入（TTM）。
        """
        # return self._to_rq_expression('cash_received_from_sales_of_goods/operating_revenue_ttm_0')
        return self.get_financial_factor('sales_service_cash_to_or_ttm_0')

    def rq_tax_ratio(self, n=0):
        """销售税金率（Tax ratio），
        计算方法：销售税金率=营业税金及附加（TTM）/营业收入（TTM）。
        """
        # return self._to_rq_expression(f'sales_tax_ttm_{n}/operating_revenue_ttm_{n}')
        return self.get_financial_factor(f'tax_ratio_ttm_{n}')

    def rq_total_profit_cost_ratio(self):
        """成本费用利润率（Total profit cost ratio）。
        计算方法：成本费用利润率=利润总额/(营业成本+财务费用+销售费用+管理费用)，以上科目使用的都是TTM的数值。
        """
        # return self._to_rq_expression('profit_before_tax_ttm_0/(cost_of_goods_sold_ttm_0+financing_expense_ttm_0+selling_expense_ttm_0+ga_expense_ttm_0)')
        return self.get_financial_factor('total_profit_cost_ratio_ttm_0')

    def rq_cfo_to_ev(self):
        """经营活动产生的现金流量净额与企业价值之比（Cash provided by operations to enterprise value）。
        """
        # return self._to_rq_expression('cash_flow_from_operating_activities_ttm_0/(market_cap_3 + total_liabilities_ttm_0)')
        return self.get_financial_factor('cfo_to_ev_ttm_0')

    def rq_acca(self):
        """现金流资产比和资产回报率之差
        """
        # return self._to_rq_expression('''((cash_flow_from_operating_activities_ttm_0 + cash_flow_from_investing_activities_ttm_0 + cash_flow_from_financing_activities_ttm_0)/total_assets_ttm_0) - return_on_asset_ttm''')
        return self.get_financial_factor('acca_ttm_0')

    def rq_ta_to_ev(self):
        """ 资产总计与企业价值之比（Assets to enterprise value）。

        - ev_ttm没有尾缀，但是可以结合公式实现尾缀, 详情查看上述formulas
        """
        return self._to_rq_expression('total_assets_ttm_0/ev_ttm')

    def rq_degm(self):
        """毛利率增长（Growth rate of gross income ratio），去年同期相比。
        计算方法：毛利率增长=(今年毛利率（TTM）/去年毛利率（TTM）)-1。
        """
        # gross_profit_margin_ttm_4 = (Factor('operating_revenue_ttm_4') - Factor('cost_of_goods_sold_ttm_4')) / Factor('operating_revenue_ttm_4')
        # return Factor('gross_profit_margin_ttm') / gross_profit_margin_ttm_4 - 1
        return self.get_financial_factor('gross_profit_margin_ttm_0_yoy')

    def rq_earnmom(self, quatars=8):
        """八季度净利润变化趋势（Change tendency of net profit in the past eight quarters）
        计算方法：前8个季度的净利润，如果同比（去年同期）增长记为+1，同比下滑记为-1，再将8个值相加。

        - 20231011：改用mrq计算，正好支持8个季度
        """
        s_mom = 0
        for q in range(quatars):
            _up = Factor(f'net_profit_mrq_{q}') > Factor(f'net_profit_mrq_{q + 4}')
            s_mom += IF(_up, 1, -1)
        return s_mom / 8


class TSFreshFactor(MyRQFactorsBase):

    @staticmethod
    def _rolling_tsfresh(series, window):
        matrix = rolling_window(series, window)

    def get_ts_method(self, func_name):
        """根据名称查询并返回指定函数"""
        try:
            func = getattr(tsfresh_calculators, func_name)
        except AttributeError:
            logger.error("TSFresh不支持公式{}", func_name)


class MACD(MyRQFactorsBase):

    def _rq_macd(self, source: Factor, fast=12, slow=26, window=9):
        """自定义MACD因子

        计算公式：
        DIFF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
        DEA = EMA(DIFF, M)
        HIST = (DIFF - DEA) * 2
        """
        # assert source in ['close', 'volume', 'total_turnover'], "目前只支持 close, volume, total_turnover"
        # if source == 'close':
        #     source = 'close_unadjusted'
        # source = Factor('close_unadjusted')
        source = Factor('close')
        diff_ = EMA(source, fast) - EMA(source, slow)
        dea_ = EMA(diff_, window)
        hist = (diff_ - dea_) * 2
        return diff_, dea_, hist

    def rq_macd_diff(self, source='close', fast=12, slow=26, window=9):
        """MACD因子-DIFF因子

        """
        # return Factor('MACD_DIFF')
        return self._rq_macd(source, fast, slow, window)[0]

    def rq_macd_dea(self, source='close', fast=12, slow=26, window=9):
        """MACD因子-DEA因子

        """
        # return Factor('MACD_DEA')
        return self._rq_macd(source, fast, slow, window)[1]

    def rq_macd_hist(self, source='close', fast=12, slow=26, window=9):
        """MACD因子-HIST因子

        """
        # return Factor('MACD_HIST')
        return self._rq_macd(source, fast, slow, window)[2]

    def rq_MACD_HIST_slp_md(self):
        """MACD_HIST的斜率的

        """
        _hist_slope36 = self._slope(Factor('MACD_HIST'), 36)
        return ABS(_hist_slope36 - MEDIAN(_hist_slope36, 30))


class BBand(MyRQFactorsBase):

    def _rq_bband(self, factor, window=20, sigma=2):
        """布林带指标计算
        计算方法：
        BOLL = MA(CLOSE, N)
        BOLLUP = BOLL + STD(CLOSE, N) * P
        BOLLDOWN = BOLL - STD(CLOSE, N) * P
        """
        _boll = MA(factor, window)
        _boll_up = _boll + STD(factor, window) * sigma
        _boll_down = _boll - STD(factor, window) * sigma
        return _boll_down, _boll, _boll_up


class MyRQVolumeFactors(MyRQFactorsBase):
    """基于成交量/成交额的简单因子"""

    def rq_vol3(self):
        return MA(self.rq_hsl(), 3)

    def rq_vol5(self):
        return MA(self.rq_hsl(), 5)

    def rq_vol120(self):
        return MA(self.rq_hsl(), 120)

    def rq_davol5(self):
        return self.rq_vol5() / self.rq_vol120()

    def rq_vrevs5(self):
        """成交量N日动量，单位：股"""
        return self._revs(Factor('volume'), window=5)

    def rq_vrevs10(self):
        return self._revs(Factor('volume'), window=10)

    def rq_vrevs20(self):
        return self._revs(Factor('volume'), window=20)

    def rq_tvrevs5(self):
        """成交额N日动量，单位：元"""
        return self._revs(Factor('total_turnover', window=5))

    def rq_tvrevs10(self):
        return self._revs(Factor('total_turnover', window=10))

    def rq_tvrevs20(self):
        return self._revs(Factor('total_turnover', window=20))

    def rq_vroc6(self):
        """成交量（股）N日变化率"""
        return self._roc(Factor('volume', window=6))

    def rq_vroc12(self):
        return self._roc(Factor('volume', window=12))

    def rq_vroc20(self):
        return self._roc(Factor('volume', window=20))

    def rq_vroc24(self):
        return self._roc(Factor('volume', window=24))

    def rq_tvroc6(self):
        """成交额（元）N日变化率"""
        return self._roc(Factor('total_turnover', window=6))

    def rq_tvroc12(self):
        return self._roc(Factor('total_turnover', window=12))

    def rq_tvroc20(self):
        return self._roc(Factor('total_turnover', window=20))

    def rq_tvroc24(self):
        return self._roc(Factor('total_turnover', window=24))

    def rq_tvrevs10_demean(self):
        """成交额（元）N日动量-去均值"""
        return self._demean(self.rq_tvrevs10, window=10)

    def rq_tvrevs20_demean(self):
        return self._demean(self.rq_tvrevs20, window=20)

    def rq_tvrevs20_divmean(self):
        """成交额（元）N日动量-除以均值"""
        return self._divmean(self.rq_tvrevs20, window=20)

    def rq_tv_div_tv120(self):
        return self.rq_turnover() / MA(self.rq_turnover(), window=120)

    def rq_tv3_div_tv120(self):
        return MA(self.rq_turnover(), window=3) / MA(self.rq_turnover(), window=120)

    def rq_tv5_div_tv120(self):
        return MA(self.rq_turnover(), window=5) / MA(self.rq_turnover(), window=120)

    def rq_tv20_div_tv120(self):
        return MA(self.rq_turnover(), window=20) / MA(self.rq_turnover(), window=120)


class MyRQPriceFactors(MyRQFactorsBase):
    """基于收盘价的简单因子"""

    def rq_skewness(self, window=20):
        """股价偏度（Skewness of price during the last N days），过去20个交易日股价的偏度。
        """
        return TS_SKEW(Factor('close'), window)

    def rq_revs5(self):
        return self._revs(Factor('close'), window=5)

    def rq_revs10(self):
        return self._revs(Factor('close'), window=10)

    def rq_revs20(self):
        return self._revs(Factor('close'), window=20)

    def rq_roc6(self):
        return self._roc(Factor('close'), window=6)

    def rq_roc12(self):
        return self._roc(Factor('close'), window=12)

    def rq_roc20(self):
        return self._roc(Factor('close'), window=20)

    def rq_roc24(self):
        return self._roc(Factor('close'), window=24)


class MyRQFactors(MyRQVolumeFactors, MyRQPriceFactors, MACD, BBand):

    def rq_dhilo(self, window=63):
        """波幅中位数（median of volatility），
        计算方法：每日对数最高价和对数最低价差值的3月内中位数。
        释义：
            “对数最高价与对数最低价的差值“其实就是最高价/最低价再取log，已经消去了股价大小的量纲；
            这个因子值越大，代表N日内最高价与最低价之间的差的中枢越大，既有可能是上涨，也有可能是下跌，因此绝对数值没有太大的意义，至少要看变化率；
            变化率可以作为另一种pct_change的度量
        """
        factor_ = LOG(Factor('high')) - LOG(Factor('low'))
        return MEDIAN(factor_, window)

    def rq_psy(self, window=12):
        """心理线指标：表示为N日内上涨日占分析周期的比例"""
        factor_ = IF(Factor('close') > REF(Factor('close'), 1), 1, 0)
        return MA(factor_, window)

    def rq_mdd_excees_return(self, window=10):
        zz1000_returns = self.rq_zz1000_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._mdd_excess_return, window,
                                           naive_returns, zz1000_returns)

    def _mdd_excess_return(self, window, naive_return, benchmark_return):
        r_naive = rolling_window(naive_return, window)
        r_benchmark = rolling_window(benchmark_return, window)
        r_ex = r_naive - r_benchmark
        r_cum_ex = np.cumprod(r_ex + 1, axis=1)
        r_cum_max_ex = np.maximum.accumulate(r_cum_ex, axis=1)
        r_ldd_ex_ret = (r_cum_ex - r_cum_max_ex) / r_cum_max_ex
        r_ldd_ex_ret = np.where(r_ldd_ex_ret < 0, r_ldd_ex_ret, 0)
        s_mdd_ex_ret = np.nanmin(r_ldd_ex_ret, axis=1)
        return s_mdd_ex_ret

    def rq_52weeks_high(self, window=240):
        """当前价格处于过去1年股价的位置,
        计算方法为个股当前价格与过去1年股价最小值之差除以过去1年股价最大值和最小值之差；
        """
        low_ = TS_MIN(Factor('low'), window)
        high_ = TS_MAX(Factor('high'), window)
        return (Factor('close') - low_) / (high_ - low_)

    def rq_illiquidity(self, window=20):
        """收益相对金额比
        计算方法： Illiquidity = sum(pct_change,20)/sum(turnover,20) * 1e9
        """
        return_ = self.rq_naive_return()
        return 1e9 * SUM(return_, window) / SUM(Factor('total_turnover'), window)

    def _rq_bias(self, window):
        """偏离率

        - 20231024：米筐含带尾缀的BIAS5，BIAS10等，这里拆解开自己计算
        """
        return ((Factor('close') - MA(Factor('close'), window)) /
                MA(Factor('close'), window) * 100)

    def rq_bias5(self):
        return self._rq_bias(5)

    def rq_bias10(self):
        return self._rq_bias(10)

    def rq_bias20(self):
        return self._rq_bias(20)

    def rq_bias60(self):
        return self._rq_bias(60)

    def _rq_biasmn(self, window1, window2):
        """M日close的移动均值与N日close的移动均值的偏离率
        """
        return MA(Factor('close'), window1) - MA(Factor('close'), window2)

    def _rq_mabiasmn(self, window1, window2, window3):
        return MA(self._rq_biasmn(window1, window2), window3)

    def rq_bias36(self):
        return self._rq_biasmn(3, 6)

    def rq_bias612(self):
        return self._rq_biasmn(6, 12)

    def rq_mabias36(self):
        return self._rq_mabiasmn(3, 6, 10)

    def rq_bband_up(self):
        """布林带-上轨

        - 20231024：米筐含该因子，未来有需要可重新计算实现window自定义，此处默认N=20，SIGMA=2
        """
        return Factor('BOLL_UP')

    def rq_bband_mid(self):
        return Factor('BOLL')

    def rq_bband_down(self):
        return Factor('BOLL_DOWN')

    def _rq_cci(self, window):
        """N日顺势指标（Commodity Channel Index）
        - 20231024：存在多个周期需要，复写原米筐生成函数
        """
        typ_ = (Factor('high') + Factor('low') + Factor('close')) / 3
        cci = (typ_ - MA(typ_, window)) / (0.015 * AVEDEV(typ_, window))
        return cci

    def rq_cci5(self):
        return self._rq_cci(5)

    def rq_cci10(self):
        return self._rq_cci(10)

    def rq_cci20(self):
        return self._rq_cci(20)

    def rq_cci88(self):
        return self._rq_cci(88)

    def rq_ema12(self):
        """
        :return:
        """
        return EMA(Factor('close'), 12)

    def rq_ema26(self):
        return EMA(Factor('close'), 26)

    def rq_kdj_k(self):
        return Factor("KDJ_K")

    def rq_kdj_d(self):
        return Factor("KDJ_D")

    def rq_kdj_j(self):
        return Factor("KDJ_J")

    # def _rq_roc(self, window):
    #     return (100 * DELTA(Factor('close'), window) /
    #             REF(Factor('close'), window))

    def rq_mtm(self, window=10):
        """动量线
        """
        return Factor('close') - REF(Factor('close'), window)

    def rq_mamtm(self, window1=10, window2=10):
        return MA(self.rq_mtm(window1), window2)

    def rq_mfi(self, window=20):
        """资金流量指标（Money Flow Index）
        TYP = (HIGH + LOW + CLOSE) / 3
        V1 = SUM(IF(TYP REF(TYPE, 1), TYP _ VOLUME, 0), N) / SUM(IF(TYP < REF(TYP, 1), TYP _ VOLUME, 0), N)
        MFI = 100 – (100 / (1+V1))
        """
        # ta.mfi
        typ_ = (Factor('high') + Factor('low') + Factor('close')) / 3
        raw_money_flow_ = typ_ * Factor('volume')
        v1_ = (SUM(IF(typ_ > REF(typ_, 1), raw_money_flow_, 0), window) /
               SUM(IF(typ_ < REF(typ_, 1), raw_money_flow_, 0), window))
        mfi_ = 100 - 100 / (1 + v1_)
        return mfi_

    def rq_mfi3(self):
        return self.rq_mfi(window=3)

    def rq_mfi6(self):
        return self.rq_mfi(window=6)

    def rq_mfi12(self):
        return self.rq_mfi(window=12)

    def rq_mfi20(self):
        return self.rq_mfi(window=20)

    def rq_mfi40(self):
        return self.rq_mfi(window=40)

    def rq_mfi60(self):
        return self.rq_mfi(window=60)

    def rq_mfi120(self):
        return self.rq_mfi(window=120)

    def _rq_dmi(self, window=12):
        """PlusDM/TR， MinusDM/TR

        TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
        HD = HIGH - REF(HIGH, 1)
        LD = REF(LOW, 1) - LOW
        DMP = SUM(IF((HD 0) & (HD LD), HD, 0), M1)
        DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
        :param window:
        :return:
        """
        # ta.dm()
        tr_ = SUM(MAX(MAX(Factor('high') - Factor('low'), ABS(Factor('high') - REF(Factor('close'), 1))), ABS(Factor('low') - REF(Factor('close'), 1))), window)

        hd_ = Factor('high') - REF(Factor('high'), 1)
        ld_ = REF(Factor('low'), 1) - Factor('low')
        dmp_ = SUM(IF((hd_ > 0) & (hd_ > ld_), hd_, 0), window)
        dmm_ = SUM(IF((ld_ > 0) & (ld_ > hd_), ld_, 0), window)
        return dmp_ * 100 / tr_, dmm_ * 100 / tr_

    def rq_dmi_dmp(self, window=12):
        return self._rq_dmi(window)[0]

    def rq_dmi_dmm(self, window=12):
        return self._rq_dmi(window)[1]

    def rq_adx(self, window1=14, window2=6):
        """ADX因子
        ADX = MA(ABS(DI2 - DI1) / (DI1 + DI2) * 100, M2)
        ADXR = (ADX + REF(ADX, M2)) / 2
        """
        di1, di2 = self._rq_dmi(window1)
        adx_ = MA(ABS(di2 - di1) / (di1 + di2) * 100, window2)
        return adx_

    def rq_adxr(self, window1=14, window2=6):
        adx_ = self.rq_adx(window1, window2)
        return (adx_ + REF(adx_, window2)) / 2

    def _rq_aroon(self, window=14):
        """阿隆指数-计算自价格达到近期最高值和最低值以来所经过的期间数
        - 20231024: 实际计算涉及argmax和argmin
        - 20231025：结果和米筐自带AROON_UP一致
        """
        # ta.aroon()
        scalar = 100
        # 20231024：注意，这里米筐自身的TS_ARGMAX和np.argmax返回值相同，所以要用window-1减去索引，这样最终值才和米筐自身的AROON_UP一致
        periods_from_hh = window - TS_ARGMAX(Factor('high'), window) - 1
        periods_from_ll = window - TS_ARGMIN(Factor('low'), window) - 1
        aroon_up = aroon_down = scalar
        aroon_up *= 1 - (periods_from_hh / window)
        aroon_down *= 1 - (periods_from_ll / window)
        aroon_osc = aroon_up - aroon_down
        return aroon_up, aroon_down, aroon_osc

    def rq_aroon_up(self, window=14):
        return self._rq_aroon(window)[0]

    def rq_aroon_down(self, window=14):
        return self._rq_aroon(window)[1]

    def rq_aroon_osc(self, window=14):
        return self._rq_aroon(window)[2]

    def _rq_trix(self, window=12):
        """N日收盘价三重指数平滑移动平均指标（Triple Exponentially Smoothed Average）
        计算方法：
        EMA3 = EMA(EMA(EMA(close, N), N), N)
        TRIX= EMA3(t) / EMA3(t-1)-1
        N取5、10等
        """
        tr_ = EMA(EMA(EMA(Factor('close'), window), window), window)
        trix_ = (tr_ - REF(tr_, 1)) / REF(tr_, 1) * 100
        return trix_

    def rq_trix5(self):
        return self._rq_trix(5)

    def rq_trix10(self):
        return self._rq_trix(10)

    def rq_matrix(self, window1=12, window2=20):
        return MA(self._rq_trix(window1), window2)

    def _rq_adtm(self, window=23):
        """动态买卖气指标
        计算方法：
        若当日开盘价大于昨日开盘价，则DTM = max(highest - open, open - prev_open) , 否则DTM=0.
        若当日开盘价小与昨日开盘价，则DBM = max(open - lowest,open- prev_open), 否则DBM=0。
        STM为N日内DTM之和，SBM为N日内DBM之和。N =20.
        ADTM = (STM - SBM) / max(STM, SBM)。
        DTM = IF(OPEN<=REF(OPEN,1),0,MAX((HIGH-OPEN),(OPEN-REF(OPEN,1))))
        DBM = IF(OPEN>=REF(OPEN,1),0,MAX((OPEN-LOW),(OPEN-REF(OPEN,1))))
        STM = SUM(DTM,N)
        SBM = SUM(DBM,N)
        ADTM = IF(STM>SBM,(STM-SBM)/STM,IF(STM=SBM,0,(STM-SBM)/SBM))
        MAADTM = MA(ADTM, M)
        """
        dtm_ = IF(Factor('open') <= REF(Factor('open'), 1), 0, MAX((Factor('high') - Factor('open')), (Factor('open') - REF(Factor('open'), 1))))
        dbm_ = IF(Factor('open') >= REF(Factor('open'), 1), 0, MAX((Factor('open') - Factor('low')), (Factor('open') - REF(Factor('open'), 1))))
        stm_ = SUM(dtm_, window)
        sbm_ = SUM(dbm_, window)
        adtm_ = IF(stm_ > sbm_, (stm_ - sbm_) / stm_, IF(stm_ == sbm_, 0, (stm_ - sbm_) / sbm_))
        return stm_, sbm_, adtm_

    def rq_stm(self, window=20):
        return self._rq_adtm(window)[0]

    def rq_sbm(self, window=20):
        return self._rq_adtm(window)[1]

    def rq_adtm(self, window=20):
        return self._rq_adtm(window)[-1]

    def rq_maadtm(self, window1=20, window2=8):
        return MA(self.rq_adtm(window1), window2)

    def rq_asi(self, window=26, ratio=True):
        """累计振动升降指标（Accumulation Swing Index），又称实质线
        计算方法：
        A = abs(close - prev_close) B = abs(lowest - prev_close) C = abs(highest - prev_lowest) D = abs(prev_close - prev_open)
        E = close - prev_close F = close - open G = prev_close - prev_open
        X = E + F / 2 + G
        K = max(A, B)
        比较A, B, C三者数值，若A最大，R=A + B / 2 + D / 4; 若B最大，R=A / 2 + B + D / 4 ; 若C最大，R = C + D / 4
        SI = 16 * X / R * K
        ASI=sum(SI,20)

        - TODOs: 精细化
        """
        LC = REF(Factor('close'), 1)
        AA = ABS(Factor('high') - LC)
        BB = ABS(Factor('low') - LC)
        CC = ABS(Factor('high') - REF(Factor('low'), 1))
        DD = ABS(LC - REF(Factor('open'), 1))
        R = IF(
            (AA > BB) & (AA > CC),
            AA + BB / 2 + DD / 4, IF(
                (BB > CC) & (BB > AA),
                BB + AA / 2 + DD / 4,
                CC + DD / 4
            )
        )
        X = (Factor('close') - LC + (Factor('close') - Factor('open')) / 2 + LC - REF(Factor('open'), 1))
        SI = X * 16 / R * MAX(AA, BB)
        asi_ = SUM(SI, window)
        if ratio:
            asi_ /= MA(Factor('close'), window)
        return asi_

    def rq_maasi(self, window1=26, window2=10, ratio=True):
        return MA(self.rq_asi(window1, ratio=ratio), window2)

    def _rq_ddi(self, window=13):
        """方向标准离差指数，观察一段时间内股价相对于前一天向上波动和向下波动的比例，并对其进行移动平均分析

        计算方法：
        若(highest+ lowest)<= (prev_highest + prev_lowest), DMZ= 0
        若(highest+ lowest) > (prev_highest + prev_lowest) , DMZ = max(abs(highest- prev_highest), abs(lowest -prev_lowest))
        若(highest + lowest) > = (prev_highest + prev_lowest) , DMF = 0
        若(highest+ lowest) < (prev_highest + prev_lowest) , DMF = max(abs(highest - prev_highest), abs(lowest -prev lowest))
        DIZ = SUM(DMZ, N) / (SUM(DMZ, N) + SUM(DMF, N))
        DIF = SUM(DMF, N) / (SUM(DMZ, N) + SUM(DMF, N))
        DDI = DIZ - DIF
        取N = 13。
        """
        high_plus_low = Factor('high') + Factor('low')
        prev_high_plus_low = REF(Factor('high'), 1) + REF(Factor('low'), 1)
        max_abs_hh_ll = MAX(
            ABS(Factor('high') - REF(Factor('high'), 1)),
            ABS(Factor('low') - REF(Factor('low'), 1))
        )
        dmz_ = IF(high_plus_low > prev_high_plus_low, max_abs_hh_ll, 0)
        dmf_ = IF(high_plus_low < prev_high_plus_low, max_abs_hh_ll, 0)
        diz_ = SUM(dmz_, window) / (SUM(dmz_, window) + SUM(dmf_, window))
        dif_ = SUM(dmf_, window) / (SUM(dmz_, window) + SUM(dmf_, window))
        ddi_ = diz_ - dif_
        return diz_, dif_, ddi_

    def rq_ddi_diz(self, window=13):
        return self._rq_ddi(window)[0]

    def rq_ddi_dif(self, window=13):
        return self._rq_ddi(window)[1]

    def rq_ddi(self, window=13):
        return self._rq_ddi(window)[2]

    def rq_ddi3(self):
        return self.rq_ddi(window=3)

    def rq_ddi6(self):
        return self.rq_ddi(window=6)

    def rq_ddi12(self):
        return self.rq_ddi(window=12)

    def rq_ddi20(self):
        return self.rq_ddi(window=20)

    def rq_ddi40(self):
        return self.rq_ddi(window=40)

    def rq_pvt(self, window=1):
        """价量趋势指标
        计算方法： PVT = (close - prev_close) / prev_close * volume 1日的累计PVT。最后入库数值除以le6.

        - 20231007：本身具有含义，不加_
        """
        pvt_ = self.rq_naive_return() * Factor('volume') / 1e9
        return MA(pvt_, window)

    def rq_pvt6(self):
        return self.rq_pvt(6)

    def rq_pvt12(self):
        return self.rq_pvt(12)

    def rq_uno(self, fast=7, median=14, slow=20):
        """终极指标（Ultimate Oscillator），
        计算方法：
        TH = max(highest, prev_close)
        TL = min(lowest, prev_close)
        TR= TH -TL
        XR =close-TL
        XRM = M日XR之和/M日TR之和
        XRN = N日XR之和/N日TR之和
        XRO = O日XR之和/O日TR之和
        UOS = 100 * (XRM * NO + XRN* MO + XRO * MN) / (MN + MO+ NO)
        """
        # ta.uo()
        th_ = MAX(Factor('high'), REF(Factor('close'), 1))
        tl_ = MIN(Factor('low'), REF(Factor('close'), 1))
        tr_ = th_ - tl_
        xr_ = Factor('close') - tl_
        xrf_ = SUM(xr_, fast) / SUM(tr_, fast)
        xrm_ = SUM(xr_, median) / SUM(tr_, median)
        xrs_ = SUM(xr_, slow) / SUM(tr_, slow)
        fast_w = 4
        median_w = 2
        slow_w = 1
        total_weights = fast_w + median_w + slow_w
        uos = 100 * (fast_w * xrf_ + median_w * xrm_ + slow_w * xrs_) / total_weights
        return uos

    def rq_srmi(self, window=10):
        """修正动量指标

        计算方法： 在当日收盘价小于前—交易日时，以前—交易日作为衡量基准；当日收盘价大于前一交易日时，以当日作为衡量基准。
        N = 10。 SRMI=(close-close(t-N))/max(close,close(t-N))
        """
        srmi = (Factor('close') - REF(Factor('close'), window)) / MAX(Factor('close'), REF(Factor('close'), window))
        return srmi

    def rq_dbcd(self, window1=5, window2=16, window3=17):
        """异同离差乖离率，先计算乖离率BIAS，然后计算不同日的乖离率之间的离差，最后对离差进行指数移动平滑处理。

        计算方法：
        BIAS = (close/ MA(close, N) - 1) * 100
        DIF = BIAS(t) - BIAS(t - M)
        DBCD = EMA(DIF, T, 1) 取N = 5, M = 16, T = 17。
        """
        bias_ = self._rq_bias(window1)
        ref_bias_ = REF(self._rq_bias(window1), window2)
        dif_ = bias_ - ref_bias_
        dbcd = EMA(dif_, window3)
        return dbcd

    def rq_arc(self, window=50):
        """变化率指数均值，股票的价格变化率RC指标的均值，用以判断前一段交易周期内股票的平均价格变化率。
        计算方法：
        ARC=EMA(RC, N, 1/N) RC=close(t)/close(t-N)
        其中N=50, 1/N为指数移动平均的加权系数α

        - FIXME: 米筐自带的EMA的平滑系数为2，如果要调整需要自定义MYEMA操作
        """
        rc_ = Factor('close') / REF(Factor('close'), window)
        arc = EMA(rc_, window)
        return arc

    def _get_factor_ad(self, order_book_ids, start_date, end_date):
        close = rqdatac.get_price(order_book_ids, start_date, end_date, fields='close', expect_df=False, adjust_type='post')
        high = rqdatac.get_price(order_book_ids, start_date, end_date, fields='high', expect_df=False, adjust_type='post')
        low = rqdatac.get_price(order_book_ids, start_date, end_date, fields='low', expect_df=False, adjust_type='post')
        volume = rqdatac.get_price(order_book_ids, start_date, end_date, fields='volume', expect_df=False, adjust_type='post')
        ad_ = close * 2 - (high + low)
        ad_ *= volume / (high - low)
        return ad_.cumsum()

    def rq_ad(self, ratio=True):
        """累积/派发线指标

        - TODOs: 精细化（因为ad同时涉及volume,price,diff，所以最后ratio处理也要涉及volume，price，diff的处理）
        - 20231123：周期改为240
        """
        ad_ = UserDefinedLeafFactor('ad', self._get_factor_ad)
        if ratio:
            ad_ /= MA(Factor('volume'), 240) / MA(Factor('close'), 240) * MA(Factor('high') - Factor('low'), 240)
        return ad_

    def rq_chosc(self, fast=3, slow=10):
        """佳庆指标(Chaikin Oscillator)。该指标基于AD曲线的指数移动均线而计算得到

        计算方法： ChaikinOscillator = EMA(AD, 3) - EMA(AD, 10)
        """
        ad_ = self.rq_ad()
        chosc = EMA(ad_, fast) - EMA(ad_, slow)
        return chosc

    def rq_chvolatility(self, window=10):
        """佳庆离散指标(Chaikin Volatility , 简称CVLT , VCI , CV)'又称“佳庆变异率指数” ，
        是通过测量一段时间内价格幅度平均值的变化来反映价格的离散程度。

        计算方法：
        HLEMA = EMA(highest - lowest, 10)
        ChaikinVolatility = 100 * (HLEMA (t) - HLEMA (t-10)) / HLEMA (t-10)
        """
        hlema_ = EMA(Factor('close') - Factor('low'), window)
        chvola = 100 * (hlema_ - REF(hlema_, window)) / REF(hlema_, window)
        return chvola

    def _rq_emv(self, window=14, ratio=True):
        """简易波动指标（Ease of Movement Value）
        计算方法：
        EMV =EMA(, N)
        """
        mv_ = (((Factor('high') + Factor('low')) / 2 - (REF(Factor('high'), 1) + REF(Factor('low'), 1)) / 2) *
               (Factor('high') - Factor('low')) / Factor('volume'))
        if ratio:
            mv_ *= MA(Factor('volume'), window)
            mv_ /= MA(Factor('close'), window)
        return EMA(mv_, window)

    def rq_emv6(self):
        return self._rq_emv(6)

    def rq_emv14(self):
        return self._rq_emv(14)

    def _rq_ulcer(self, window=10):
        """
        用于考察向下的波动性
        计算方法： Ri= (close-max(close,n))/max(close,n)
        """
        ri_ = Factor('close') / TS_MAX(Factor('close'), window) - 1
        ri_ *= 100
        return SIGNEDPOWER(MA(SIGNEDPOWER(ri_, 2), window), 0.5)

    def rq_ulcer5(self):
        return self._rq_ulcer(5)

    def rq_ulcer10(self):
        return self._rq_ulcer(10)

    def rq_apbma(self, window=12, ratio=True):
        """绝对偏差移动平均（Absolute Price Bias Moving Average）。考察一段时期内价格偏离均线的移动平均。

        计算方法： APBMA = MA(abs(close - MA(close, N)), N)
        """
        apb_ = ABS(Factor('close') - MA(Factor('close'), window))
        if ratio:
            apb_ /= MA(Factor('close'), window)
        return MA(apb_, window)

    def rq_bbi(self, ratio=True):
        """多空指数（Bull and Bear Index）。是一种将不同日数移动平均线加权平均之后的综合指标

        计算方法：BBI = (MA3 + MA6 + MA12 + MA24) / 4

        - 20231025：bbi精细化处理时已经去掉量纲，所以不再需要bbic=self.rq_bbi() / Factor('close')
        - TODOs: 精细化
        """
        ma3_ = MA(Factor('close'), 3)
        ma6_ = MA(Factor('close'), 6)
        ma12_ = MA(Factor('close'), 12)
        ma24_ = MA(Factor('close'), 24)
        bbi = (ma3_ + ma6_ + ma12_ + ma24_) / 4
        if ratio:
            bbi /= Factor('close')
            # bbi /= MA(Factor('close'), 240)
        return bbi

    def rq_bbi_ratioma(self, window=30, ratio=True):
        ma3_ = MA(Factor('close'), 3)
        ma6_ = MA(Factor('close'), 6)
        ma12_ = MA(Factor('close'), 12)
        ma24_ = MA(Factor('close'), 24)
        bbi = (ma3_ + ma6_ + ma12_ + ma24_) / 4
        if ratio:
            bbi /= MA(Factor('close'), window)
        return bbi

    def rq_bbi_ratioma20(self):
        return self.rq_bbi_ratioma(20)

    def rq_bbi_ratioma10(self):
        return self.rq_bbi_ratioma(10)

    def _rq_tema(self, window=10, ratio=True):
        """计算方法：
        TEMA = 3 * EMA(close, N) - 3 * EMA(EMA(close, N), N) + EMA(EMA(EMA(close, N), N), N)
        """
        ema_ = EMA(Factor('close'), window)
        eema_ = EMA(ema_, window)
        eeema_ = EMA(eema_, window)
        tema_ = 3 * ema_ - 3 * eema_ + eeema_
        if ratio:
            tema_ /= Factor('close')
        return tema_

    def rq_tema5(self):
        return self._rq_tema(5)

    def rq_tema10(self):
        return self._rq_tema(10)

    def rq_maratio(self, window=10):
        return MA(Factor('close'), window) / Factor('close')

    def rq_massindex(self, window1=25, window2=9):
        """梅斯线(Mass Index)
        计算方法：
        EMAHL = EMA(highest - lowest, 9)。
        EMA Ratio= EMAHL / EMA(EMAHL, 9)。
        Masslndex = EMA Ratio的25天的累加值。
        """
        hlema_ = EMA(Factor('high') - Factor('low'), window2)
        hlema_ratio_ = hlema_ / EMA(hlema_, window2)
        massindex_ = SUM(hlema_ratio_, window1)
        return massindex_

    def rq_bearpower(self, window=13, ratio=True):
        """空头力道
        计算方法: BearPower = lowest - EMA(close, N), 其中N取13。
        """
        bearp_ = Factor('low') - EMA(Factor('close'), window)
        if ratio:
            bearp_ /= MA(Factor('close'), window)
        return bearp_

    def rq_bullpower(self, window=13, ratio=True):
        """多头力道
        计算方法： Bull Power = highest - EMA(close, N), 其中N取13.
        """
        bullp_ = Factor('high') - EMA(Factor('close'), window)
        if ratio:
            bullp_ /= MA(Factor('close'), window)
        return bullp_

    def rq_elder(self, window=13):
        """艾达透视指标(Elder-ray Index)
        计算方法：
        BullPower = highest - EMA(close, N)。
        BearPower = lowest - EMA(close, N)。
        Elder = (Bull Power - BearPower) / close; N
        """
        _bearpower = self.rq_bearpower(window, ratio=False)
        _bullpower = self.rq_bullpower(window, ratio=False)
        return (_bearpower - _bullpower) / Factor('close')

    def rq_jdqs(self, window=20, benchmark_order_book_id='000905.XSHG'):
        """
        计算方法:

        A= (N天中大盘收阴线，个股收阳线的天数)。
        B = (N天中大盘收阴线的天数)。
        JDQS =A/ B
        N = 20。
        """
        index_return_ = self._rq_benchmark_return(benchmark_order_book_id)
        naive_return = self.rq_naive_return()
        a_ = IF((index_return_ < 0) & (naive_return > 0), 1, 0)
        b_ = IF(index_return_ < 0, 1, 0)
        return SUM(a_, window) / SUM(b_, window)

    def _rq_rvi(self, window1=10, window2=14):
        """相对离散指数（Relative Volatility Index），又称“相对波动性指标”，用于测量价格的发散趋势，
        计算每日收盘价标准差SD, Dorsey建议向前取10个交易日区间计算。
        若当日收盘价大于昨日收盘价，当日记为上升日，USO = SD , DSD = 0 ;
        若当日收盘价小于昨日收盘价，当日记为下降日，USD = 0 , DSD = SD.
        对过去一段时间内的上升日和下降日的标准差求N日Wilder's Smoothfng ,
        Dorsey建议向前取14个交易日（计算2N-1日的EMA将得到相同结果，但是速度更快).
        UpRVI = EMA(USD, 2N-1) ,
        DownRVI = EMA(DSD, 2N-1) ;
        RVI = 100 * UpRVI / (UpRVI + DownRVI)。
        """
        source_ = Factor('close')  # 未来可以改为high，low，以及三者的均值等等，参考ta.rvi()的处理

        std_ = STD(source_, window1)
        # 这里为了避免存在超过window2周期的连续上涨或下跌导致中间变量为int:0，无法参与后续计算，这里生成一个0因子
        zeros_ = self._rq_zeros()
        upstd_ = IF(source_ > REF(source_, 1), std_, zeros_)
        dnstd_ = IF(source_ < REF(source_, 1), std_, zeros_)
        uprvi_ = EMA(upstd_, window2)
        dnrvi_ = EMA(dnstd_, window2)

        rvi_ = 100 * uprvi_ / (uprvi_ + dnrvi_)
        return uprvi_, dnrvi_, rvi_

    def rq_rvi_up(self, window1=10, window2=14):
        return self._rq_rvi(window1, window2)[0]

    def rq_rvi_down(self, window1=10, window2=14):
        return self._rq_rvi(window1, window2)[1]

    def rq_rvi(self, window1=10, window2=14):
        return self._rq_rvi(window1, window2)[2]

    def rq_cmo(self, window=14):
        """钱德动量摆动指标（Chande Momentum Osciliator）
        与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。
        计算方法
        1、SU是今日收盘价与昨日收盘价（上涨日）差值加总。
        若当日下跌，则增加值为0；
        2、SD是今日收盘价与昨日收盘价（下跌日）差值的绝对值加总。
        若当日上涨，则增加值为0。
        3、CMO = (SU - SD)/ (SU + SD) * 100.
        """
        # ta.cmo()
        source_ = Factor('close')
        diff_ = DELTA(source_, 1)
        su_ = SUM(IF(source_ > REF(source_, 1), ABS(diff_), 0), window)
        sd_ = SUM(IF(source_ < REF(source_, 1), ABS(diff_), 0), window)
        cmo_ = 100 * (su_ - sd_) / (su_ + sd_)
        return cmo_

    def _rq_kvo(self, fast=34, slow=55, signal=13, ratio=True):
        """成交量摆动指标，该指标在决定长期资金流量趋势的同时保持了对于短期资金流量的敏感性，因而可以用于预测短期价格拐点

        """
        # ta.kvo()
        # TODOs: 这里的signed_series有细微的差异在于起始值是NaN而非1
        hlc = (Factor('high') + Factor('low') + Factor('close')) / 3
        signed_hlc = IF(DELTA(hlc, 1) > 0,
                        1, IF(DELTA(hlc, 1) < 0, -1, 0))
        sv = Factor('volume') * signed_hlc
        if ratio:
            # sv /= (MA(hlc, slow) * MA(Factor('volume'), slow))
            sv /= (MA(Factor('volume'), slow))  # 分布会更符合正态一点

        kvo = EMA(sv, fast) - EMA(sv, slow)
        kvo_signal = EMA(kvo, signal)
        return kvo, kvo_signal

    def rq_kvo(self, fast=34, slow=55, signal=13):
        return self._rq_kvo(fast, slow, signal)[0]

    def rq_kvo_signal(self, fast=34, slow=55, signal=13):
        return self._rq_kvo(fast, slow, signal)[1]

    def rq_coppock(self, window=20, fast=7, slow=14):
        """估波指标（Coppock Curve），又称“估波曲线”，该指标通过计算月度价格的变化速率的加权平均值来测量市场的动量，
        """
        # ta.coppock()
        # source = Factor('close')
        total_roc = self._rq_roc(fast) + self._rq_roc(slow)
        coppock_ = WMA(total_roc, window)
        return coppock_

    @staticmethod
    def _rolling_ts_regress(series, window):
        x = np.arange(1, window + 1)  # 构建时序排序x，从小到大，越靠近当前的日期值越大，此时回归系数表明越靠近当前日期的因子值的影响
        X = np.column_stack([np.ones_like(x), x])  # 构建X矩阵，对所有都是一样的1, window的值
        Y = rolling_window(series, window).T  # Y 就是移动窗口，转置是为了符合习惯性思维（行为样本数，列为不同组样本）
        B = np.linalg.pinv((X.T).dot(X)).dot(X.T).dot(Y)
        # 这里核查了返回的系数顺序没问题，即第一个系数是Y的第一列的系数，也就是转置前第一行的系数，也就是最早的日期的系数
        return B[1]  # 0行是常数项系数，1行是x系数

    def _rq_rolling_ts_regress(self, factor, window, ratio=True):
        coeff_ = RollingWindowFactor(self._rolling_ts_regress, window, factor)
        if ratio:
            coeff_ /= factor
        return coeff_

    def rq_ma10_ts_reg_coeff6(self):
        """10日价格平均线6日线性回归系数"""
        return self._rq_rolling_ts_regress(MA(Factor('close'), 10), 6)

    def rq_ma10_ts_reg_coeff12(self):
        """10日价格平均线12日线性回归系数"""
        return self._rq_rolling_ts_regress(MA(Factor('close'), 10), 12)

    def rq_plrc6(self):
        """6日收盘价格线性回归系数"""
        return self._rq_rolling_ts_regress(Factor('close'), 6)

    def rq_plrc12(self):
        """12日收盘价格线性回归系数"""
        return self._rq_rolling_ts_regress(Factor('close'), 12)

    def rq_dvrat(self, window=490):
        """收益相对波动（Daily returns variance ratio-serial dependence in daily returns）。

        - TODOs：如何指定最小窗口min_period=180
        - 老版本-定义滑动窗口实现
        # def sigma_square_q(series, window):
        #     q = 10
        #     m = q * (window - q + 1) * (1 - q / window)
        #     # sigma = np.power(np.nansum(rolling_window(series, q), axis=1), 2)
        #     sigma = np.power(self.rolling_sum(series, q), 2)
        #     # sigma_q = np.nansum(rolling_window(sigma, window - q + 1), axis=1) / m
        #     sigma_q = self.rolling_sum(sigma, window - q + 1) / m
        #     sigma_s = np.nanvar(rolling_window(series, window), axis=1)
        #     return sigma_q / sigma_s - 1
        # return RollingWindowFactor(sigma_square_q, window, ex_ret)
        """
        ex_ret = self.rq_excess_return()

        q = 10
        m = q * (window - q + 1) * (1 - q / window)
        sigma_q_tmp = ABS(SIGNEDPOWER(SUM(ex_ret, q), 2))
        sigma_q = SUM(sigma_q_tmp, window - q + 1) / m
        sigma_s = VAR(ex_ret, window)
        dvrat_ = sigma_q / sigma_s - 1
        return dvrat_

    def rq_dvrat120(self):
        return self.rq_dvrat(window=120)

    def rq_dvrat240(self):
        return self.rq_dvrat(window=240)

    def rq_dvrat480(self):
        return self.rq_dvrat(window=480)

    def rq_ar(self, window=26):
        """人气指标（AR）

        计算方法：
        AR=sum((highest-open),N)/sum((open-lowest),N) N=26
        """
        ar_ = SUM(Factor('high') - Factor('open'), window) / SUM(Factor('open') - Factor('low'), window)
        return ar_

    def rq_br(self, window=26):
        """意愿指标（BR）

        计算方法：
        BR=sum(max(highest - Delay(close,1),0),N)/sum(max(Delay(close,1) - lowest,0),N) N=26
        """
        _f_hmpc = MAX(Factor('high') - REF(Factor('close'), 1), 0)
        _hmpc = SUM(_f_hmpc, window)
        _f_pcml = MAX(REF(Factor('close'), 1) - Factor('low'), 0)
        _pcml = SUM(_f_pcml, window)
        _br = _hmpc / _pcml
        return _br

    def rq_arbr(self, window=26):
        return self.rq_ar(window) - self.rq_br(window)

    def rq_br_div_close(self):
        _br = self.rq_br()
        ma26 = MA(Factor('close'), 26)
        return _br / ma26

    def rq_ar_div_close(self):
        return self.rq_ar() / MA(Factor('close'), 26)

    def rq_cr(self, window=20):
        """N日的CR指标

        计算方法：
        TYP = (highest + lowest + close)/ 3.
        CR = sum(max(highest - prev_typical , 0), N) / sum(max(prev_typical - lowest, 0), N) * 100 取N = 20 。
        """
        typ_ = (Factor('high') + Factor('low') + Factor('close')) / 3
        cr_ = (SUM(MAX(Factor('high') - REF(typ_, 1), 0), window) /
               SUM(MAX(REF(typ_, 1) - Factor('low'), 0), window))
        return cr_

    def rq_vr(self, window=24, ratio=True):
        """成交量比率（Volume Ratio），通过分析股价上升日成交额（或成交量，下同）与股价下降日成交额比值，从而掌握市场买卖气势的中期技术指标

        计算方法：
        对任一交易日，若close > prev_close , 当日成交量为AV, 否则AV=0。将N日内AV加和后记为AVS.
        对任一交易日，若close< prev_close , 当日成交量为BV , 否则BV= 0。 将N日内BV加和后记为BVS。
        对任一交易日，若close= prev_close , 当日成交量为CV, 否则CV= 0。 将N 日内CV加和后记为CVS。
        VR = (AVS + CVS / 2) / (BVS + CVS / 2)。 N取24, 即24个交易日。
        """
        zeros = self._rq_zeros()
        if ratio:
            base_factor = self.rq_turnover()
        else:
            base_factor = Factor('volume')
        av = IF(Factor('close') > REF(Factor('close'), 1), base_factor, 0)
        bv = IF(Factor('close') < REF(Factor('close'), 1), base_factor, 0)
        cv = IF(Factor('close') == REF(Factor('close'), 1), base_factor, zeros)

        avs = SUM(av, window)
        bvs = SUM(bv, window)
        cvs = SUM(cv, window)

        return (avs + cvs / 2) / (bvs + cvs / 2)

    def rq_vr_absolute(self):
        return self.rq_vr(ratio=False)

    def _rq_acd(self, window=6, ratio=True):
        """收集派发指标
        计算方法：
        若当日收盘价高于昨日收盘价，则收集力量等于当日收盘价与真实低位之差。
        真实低位是当日低位与昨日收盘价两者中较低者。 buy= close - min(lowest, prev_close)
        若当日收盘价低于昨日收盘价，则派发力量等于当日收盘价与真实高位之差。
        真实高位是当日高位与昨日收盘价两者中较高者。 sell = close - max(highest, prev_close)
        将收集力量( buy ， 正数）及派发力量( sell , 负数）相加，即可得到市场的净收集力量ACD.
        ACD = sum(buy) + sum(sell)。

        -TODOs: 去量纲
        """
        _buy = Factor('close') - MIN(Factor('low'), REF(Factor('close'), 1))
        _sell = Factor('close') - MAX(Factor('high'), REF(Factor('close'), 1))
        mean_close = MA(Factor('close'), window)
        _acd = SUM(_buy + _sell, window)

        if ratio:
            return _acd / mean_close
        else:
            return _acd

    def rq_acd6(self, ):
        return self._rq_acd(6)

    def rq_acd12(self, ):
        return self._rq_acd(12)

    def rq_acd6_absolute(self):
        return self._rq_acd(6, False)

    def rq_acd12_absolute(self):
        return self._rq_acd(12, False)

    def rq_zz500_hv(self, window=60):
        zz500_ret = self.rq_zz500_return()
        return STD(zz500_ret, window) * math.sqrt(240 / window)

    def rq_zz500_mahv(self, window1=60, window2=20):
        return MA(self.rq_zz500_hv(window1), window2)

    def _cr(self, window, naive_return, benchmark_return, side='down'):
        # rolling_window返回的是一个矩阵，其中每一行是一组window长的值，一共有N-window+1行。
        r_naive = rolling_window(naive_return, window)
        # print("纯收益率", len(naive_return))
        # print("基准", len(benchmark_return))
        r_benchmark = rolling_window(benchmark_return, window)
        if side == 'down':
            r_naive_dd = np.where(r_benchmark < 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark < 0, r_benchmark, np.NaN)
        elif side == 'up':
            r_naive_dd = np.where(r_benchmark > 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark > 0, r_benchmark, np.NaN)
        else:
            raise ValueError("side参数只支持down和up。")

        n_rows = r_naive.shape[0]
        coefs = []
        for i in range(n_rows):
            s_naive_dd = r_naive_dd[i, :]
            s_naive_dd = s_naive_dd[~np.isnan(s_naive_dd)]
            s_benchmark_dd = r_benchmark_dd[i, :]
            s_benchmark_dd = s_benchmark_dd[~np.isnan(s_benchmark_dd)]
            # coef = pearsonr(s_naive_dd, s_benchmark_dd)[0]  # scipy.stats.pearsonr太慢了，查看底层代码主要是因为其有很多额外的判断
            coef = np.corrcoef(s_naive_dd, s_benchmark_dd)[0, 1]  # np原生更快，且两者结果一样（np.corrcoef返回的也是Pearson Correlation）
            coefs.append(coef)
        return coefs

    def _ddncr(self, window, naive_return, benchmark_return):
        """
        DDNCR：下跌相关系数（Downside correlation），
        - 过往12个月中，市场组合日收益为负时，个股日收益关于市场组合日收益的相关系数

        :param window: 窗口长度
        :param naive_return: 个股日度收益率序列
        :param benchmark_return: 基准市场组合收益率序列
        :return: 列表形式，长度为N-window+1，其中N为样本长度（若有效值<window以NaN替代）
        """
        return self._cr(window, naive_return, benchmark_return, 'down')

    def _uupcr(self, window, naive_return, benchmark_return):
        return self._cr(window, naive_return, benchmark_return, 'up')

    def rq_ddncr(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._ddncr, window,
                                           naive_returns, zz500_returns)

    def rq_ddncr60(self):
        return self.rq_ddncr(window=60)

    def rq_ddncr120(self):
        return self.rq_ddncr(window=120)

    def rq_ddncr240(self):
        return self.rq_ddncr(window=240)

    def rq_uupcr(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._uupcr, window,
                                           naive_returns, zz500_returns)

    def rq_uupcr60(self):
        return self.rq_uupcr(window=60)

    def rq_uupcr120(self):
        return self.rq_uupcr(window=120)

    def rq_uupcr240(self):
        return self.rq_uupcr(window=240)

    def _bt(self, window, naive_return, benchmark_return, side='down'):
        r_naive = rolling_window(naive_return, window)
        r_benchmark = rolling_window(benchmark_return, window)
        if side == 'down':
            r_naive_dd = np.where(r_benchmark < 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark < 0, r_benchmark, np.NaN)
        elif side == 'up':
            r_naive_dd = np.where(r_benchmark > 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark > 0, r_benchmark, np.NaN)
        else:
            raise ValueError("side只支持down和up。")

        n_rows = r_naive.shape[0]
        betas = []
        for i in range(n_rows):
            s_naive_dd = r_naive_dd[i, :]
            s_naive_dd = s_naive_dd[~np.isnan(s_naive_dd)]
            s_benchmark_dd = r_benchmark_dd[i, :]
            s_benchmark_dd = s_benchmark_dd[~np.isnan(s_benchmark_dd)]
            # polyfit(x, y, degree) polyfit默认含常数项进行回归
            beta = np.polyfit(s_benchmark_dd, s_naive_dd, 1)[0]
            betas.append(beta)
        return betas

    def _ddnbt(self, window, naive_return, benchmark_return):
        """
        DDNBT：下跌贝塔（Downside beta），
        - 过往12个月中，市场组合日收益为负时，个股日收益关于市场组合日收益的回归系数。属于收益和风险类因子。

        :param window: 窗口长度
        :param naive_return: 个股日度收益率序列
        :param benchmark_return: 基准市场组合收益率序列
        :return: 列表形式，长度为N-window+1，其中N为样本长度（若有效值<window以NaN替代）
        """
        return self._bt(window, naive_return, benchmark_return, 'down')

    def _uupbt(self, window, naive_return, benchmark_return):
        return self._bt(window, naive_return, benchmark_return, 'up')

    def rq_ddnbt(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._ddnbt, window,
                                           naive_returns, zz500_returns)

    def rq_ddnbt60(self):
        return self.rq_ddnbt(window=60)

    def rq_ddnbt120(self):
        return self.rq_ddnbt(window=120)

    def rq_ddnbt240(self):
        return self.rq_ddnbt(window=240)

    def rq_uupbt(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._uupbt, window,
                                           naive_returns, zz500_returns)

    def rq_uupbt60(self):
        return self.rq_uupbt(window=60)

    def rq_uupbt120(self):
        return self.rq_uupbt(window=120)

    def rq_uupbt240(self):
        return self.rq_uupbt(window=240)

    def _sr(self, window, naive_return, benchmark_return, side='down'):
        """
        下跌波动（Downside standard deviations ratio），
        - 过往12个月中，市场组合日收益为负时，个股日收益标准差和市场组合日收益标准差之比。属于收益和风险类因子。

        :param window: 窗口长度
        :param naive_return: 个股日度收益率序列
        :param benchmark_return: 基准市场组合收益率序列
        :return: 列表形式，长度为N-window+1，其中N为样本长度（若有效值<window以NaN替代）
        """
        r_naive = rolling_window(naive_return, window)
        r_benchmark = rolling_window(benchmark_return, window)
        if side == 'down':
            r_naive_dd = np.where(r_benchmark < 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark < 0, r_benchmark, np.NaN)
        elif side == 'up':
            r_naive_dd = np.where(r_benchmark > 0, r_naive, np.NaN)
            r_benchmark_dd = np.where(r_benchmark > 0, r_benchmark, np.NaN)
        else:
            raise ValueError("只支持down和up。")

        std_naive = np.nanstd(r_naive_dd, axis=1)
        std_benchmark = np.nanstd(r_benchmark_dd, axis=1)
        return std_naive / std_benchmark

    def _ddnsr(self, window, naive_return, benchmark_return):
        """
        下跌波动（Downside standard deviations ratio），
        - 过往12个月中，市场组合日收益为负时，个股日收益标准差和市场组合日收益标准差之比。属于收益和风险类因子。

        :param window: 窗口长度
        :param naive_return: 个股日度收益率序列
        :param benchmark_return: 基准市场组合收益率序列
        :return: 列表形式，长度为N-window+1，其中N为样本长度（若有效值<window以NaN替代）
        """
        return self._sr(window, naive_return, benchmark_return, 'down')

    def _uupsr(self, window, naive_return, benchmark_return):
        return self._sr(window, naive_return, benchmark_return, 'up')

    def rq_ddnsr(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._ddnsr, window,
                                           naive_returns, zz500_returns)

    def rq_ddnsr60(self):
        return self.rq_ddnsr(window=60)

    def rq_ddnsr120(self):
        return self.rq_ddnsr(window=120)

    def rq_ddnsr240(self):
        return self.rq_ddnsr(window=240)

    def rq_uupsr(self, window=240):
        zz500_returns = self.rq_zz500_return()
        naive_returns = self.rq_naive_return()
        return CombinedRollingWindowFactor(self._uupsr, window,
                                           naive_returns, zz500_returns)

    def rq_uupsr60(self):
        return self.rq_uupsr(window=60)

    def rq_uupsr120(self):
        return self.rq_uupsr(window=120)

    def rq_uupsr240(self):
        return self.rq_uupsr(window=240)

    def _tobt(self, window, naive_return, benchmark_return, turnover):
        """超额流动（Liquidity-turnover beta）。属于收益和风险类因子。
        - window为500，但计算时需要添加5个滞后项，保证最后500个样本非NaN，当个股总样本长度不足505时会报错
        - naive_return, benchmark_return 都是减去无风险收益率的超额收益率
        """
        r_naive_abs = np.abs(rolling_window(naive_return, window + 5))
        r_benchmark_abs = np.abs(rolling_window(benchmark_return, window + 5))
        r_turnover = rolling_window(turnover, window + 5)

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
                beta = np.linalg.pinv((X.T).dot(X)).dot(X.T).dot(y)[1]  # 第0项是常数项，第1项是所求系数
            except LinAlgError:
                beta = np.NaN
            betas.append(beta)
        return betas

    def rq_tobt(self, window=480):
        # 这里自定义的zz500的超额收益必须写在新的因子中，而不是两个自定义因子相减，
        zz500_excess_returns = self.rq_zz500_excess_return()
        excess_returns = self.rq_excess_return()
        turnover_ratio = self.rq_turnover()
        return CombinedRollingWindowFactor(self._tobt, window,
                                           excess_returns, zz500_excess_returns, turnover_ratio)

    def rq_tobt480(self):
        return self.rq_rsrs(window=480)

    def rq_tobt240(self):
        return self.rq_rsrs(window=240)

    def rq_tobt120(self):
        return self.rq_rsrs(window=120)

    def rq_accer(self, window=8):
        return SLOPE(Factor('close'), window) / Factor('close')

    def rq_accer3(self):
        return self.rq_accer(window=3)

    def rq_accer6(self):
        return self.rq_accer(window=6)

    def rq_accer12(self):
        return self.rq_accer(window=12)

    def rq_accer20(self):
        return self.rq_accer(window=20)

    def rq_accer40(self):
        return self.rq_accer(window=40)

    def rq_accer60(self):
        return self.rq_accer(window=60)

    def rq_accer120(self):
        return self.rq_accer(window=120)

    def rq_cyf(self, window=21):
        """市场能量指标（CYF）

        使用原则：把握好CYF指标一冲天价调头；二摸天价背离；CYF指标与股价同跌，则可以完全避免被股市深套
        """
        _hsl = self.rq_turnover()
        return 100 * (1 - 1 / (1 + EMA(_hsl, window)))

    def rq_swl(self, window1=5, window2=3):
        close = Factor('close')
        _swl = (EMA(close, window1) * 7 + EMA(close, window2) * 3) / 10
        return _swl

    def rq_sws(self, window=12):
        close = Factor('close')
        total_turnover = Factor('total_turnover')
        mcap = Factor('market_cap_2')
        # 这里与原公式稍有不同，将volume-capital改为成交额(total_turnover)-流通市值(mcap)，本质没有差异
        _sws = DMA(EMA(close, window), MAX(1, 100 * SUM(total_turnover, 5) / (3 * mcap)))
        return _sws

    def rq_fsl(self):
        """分水岭指标（FSL）

        使用原则：SWL>SWS算强势，否则弱势；当股价在FSL上方运行算强势，否则弱势.
                这里为了便于使用，导出时暂时使用
        计算公式：
        SWL = (EMA(CLOSE,5)7+EMA(CLOSE,10)3)/10
        SWS = DMA(EMA(CLOSE,12),MAX(1,100(SUM(VOLUME,5)/(3CAPITAL))))
        """
        _swl = self.rq_swl()
        _sws = self.rq_sws()
        return (_swl + _sws) / 2

    def rq_dkx(self):
        """多空线

        计算方法：原公式最后一个是20，但逻辑上应该是19，这里改为19
        MID = (3CLOSE+LOW+OPEN+HIGH)/6
        DKX = (20MID+19REF(MID,1)+18REF(MID,2)+17REF(MID,3)+16REF(MID,4)+15REF(MID,5)+14REF(MID,6)+
        13REF(MID,7)+12REF(MID,8)+11REF(MID,9)+10REF(MID,10)+9REF(MID,11)+8REF(MID,12)+7REF(MID,13)+
        6REF(MID,14)+5REF(MID,15)+4REF(MID,16)+3REF(MID,17)+2REF(MID,18)+REF(MID,19))/210
        MADKX = MA(DKX, M)
        """
        _mid = (3 * Factor('close') + Factor('open') + Factor('high')) / 6
        _dkx = 0
        for i in range(20):
            _v = (20 - i) * REF(_mid, i)
            _dkx += _v
        return _dkx


class WorldQuantAlpha101(MyRQFactorsBase):

    def rq_alpha001(self):
        return Factor('WorldQuant_alpha001')


# 动态添加方法到已存在的类中
def create_alpha_method(num):
    def alpha_method(self):
        return Factor(f'WorldQuant_alpha{str(num).zfill(3)}')

    return alpha_method


for i in range(2, 102):
    method_name = f'rq_alpha{str(i).zfill(3)}'
    method = create_alpha_method(i)
    setattr(WorldQuantAlpha101, method_name, method)


class Turnover(MyRQFactors):

    def rq_turnover_volatility(self, window=20):
        """换手率移动窗口波动率（Volatility of daily turnover during the last N days）"""
        turnover_ = self.rq_turnover()
        return STD(turnover_, window)

    def rq_relative_turnover_volatility(self, window=20):
        """换手率相对波动率（Relative volatility of daily turnover during the last N days）"""
        turnover_ = self.rq_turnover()
        return STD(turnover_, window) / MA(turnover_, window)


class NPVI(MyRQFactors):

    def _get_factor_nvi(self, order_book_ids, start_date, end_date):
        roc_ = 100 * rqdatac.get_price_change_rate(order_book_ids, start_date, end_date)
        volume_ = rqdatac.get_price(order_book_ids, start_date, end_date, fields='volume', expect_df=False)

        signed_volume = self._signed_series(volume_, 1)
        nvi = signed_volume[signed_volume < 0].abs() * roc_
        nvi.fillna(0, inplace=True)
        nvi.iloc[0] = 100
        nvi = nvi.cumsum()
        return nvi.reindex(columns=order_book_ids)

    def rq_nvi(self):
        """负成交量（价格）指数
        NVI=NV+100*（close-prev_close）/close × NV if volume < prev_volume
        - 基于pandas_ta.nvi构建，NV起始值=100
        - 起点日期的选择影响当期指标值。
        - ta.nvi()
        """
        return UserDefinedLeafFactor('nvi', self._get_factor_nvi)

    def _get_factor_pvi(self, order_book_ids, start_date, end_date):
        roc_ = 100 * rqdatac.get_price_change_rate(order_book_ids, start_date, end_date)
        volume_ = rqdatac.get_price(order_book_ids, start_date, end_date, fields='volume', expect_df=False)

        signed_volume = self._signed_series(volume_, 1)
        pvi = signed_volume[signed_volume > 0].abs() * roc_
        pvi.fillna(0, inplace=True)
        pvi.iloc[0] = 100
        pvi = pvi.cumsum()
        return pvi.reindex(columns=order_book_ids)

    def rq_pvi(self):
        """正成交量（价格）指数
        PVI=PV+100*（CLS－CLSn）/CLSn×PV if volume increases
        - 基于pandas_ta.pvi构建，NV起始值=100
        - FIXMEd：起点值影响指标值。
        """
        return UserDefinedLeafFactor('pvi', self._get_factor_pvi)

    def rq_pvi20_deri(self):
        """PVI差值的绝对偏离度
        - 评价：整体都还可以，尤其在区分最差组效果不错，2023分层明显，参数大小影响不大
        """
        _pvi20 = DELTA(self.rq_pvi(), 252)
        return ABS(_pvi20 - MA(_pvi20, 20))

    def rq_pvid252_z504(self):
        _delta_pvi = DELTA(self.rq_pvi(), 240)
        return TS_ZSCORE(_delta_pvi, 240)

    def rq_nvi20_deri(self):
        """NVI差值的绝对偏离度
        - 评价：整体都还可以，尤其在区分最差组效果不错，2023分层明显，参数大小影响不大
        """
        _nvi20 = DELTA(self.rq_nvi(), 240)
        return ABS(_nvi20 - MA(_nvi20, 20))

    def rq_nvid240_z504(self):
        _delta_nvi = DELTA(self.rq_nvi(), 240)
        return TS_ZSCORE(_delta_nvi, 490)

    def rq_pvid240_z504(self):
        _delta_pvi = DELTA(self.rq_pvi(), 240)
        return TS_ZSCORE(_delta_pvi, 490)


class RSRS(MyRQFactors):

    def rq_rsrs(self, window=20):
        """RSRS：阻力支撑相对强度

        计算方法：N日最高价和最低价的线性回归的系数
        """
        high_ = Factor('high')
        low_ = Factor('low')
        betas_ = TS_REGRESSION(high_, low_, window)
        return betas_ - 1

    def rq_rsrs10(self):
        return self.rq_rsrs(window=10)

    def rq_rsrs20(self):
        return self.rq_rsrs(window=20)

    def rq_rsrs18(self):
        return self.rq_rsrs(window=18)

    def rq_rsrs40(self):
        return self.rq_rsrs(window=40)

    def rq_rsrs_abs(self, window=20):
        return ABS(self.rq_rsrs(window))

    def rq_rsrs_std(self, window1=20, window2=490):
        """RSRS-标准分

        计算方法：RSRS指标取zscore
        """
        return self._zscore(self.rq_rsrs(window1), window2)

    def rq_rsrs_std10(self):
        return self.rq_rsrs_std(window1=10, window2=450)

    def rq_rsrs_std20(self):
        return self.rq_rsrs_std(window1=20, window2=450)

    def rq_rsrs_std40(self):
        return self.rq_rsrs_std(window1=40, window2=450)

    def rq_rsrs_radjust(self, window1=20, window2=504):
        """RSRS-右偏标准分

        计算方法：RSRS原值*RSRS标准分
        """
        _rsrs = self.rq_rsrs(window1)
        _rsrs_std = self.rq_rsrs_std(window1, window2)
        return (_rsrs + 1) * _rsrs_std

    def rq_rsrs_zz500_abs(self, window=20, threshold=0.2):
        _zz500_mahv = self.rq_zz500_mahv()
        _rsrs = self.rq_rsrs(window)
        _rsrs_abs = ABS(_rsrs)
        _rsrs_zz500_abs = IF(_zz500_mahv <= threshold, _rsrs_abs, _rsrs)
        return _rsrs_zz500_abs

    def rq_rsrs10_zz500_abs(self):
        return self.rq_rsrs_zz500_abs(window=10, threshold=0.2)

    def rq_rsrs20_zz500_abs(self):
        return self.rq_rsrs_zz500_abs(window=20, threshold=0.2)

    def rq_rsrs40_zz500_abs(self):
        return self.rq_rsrs_zz500_abs(window=40, threshold=0.2)


class RSI(MyRQFactors):

    def rq_rsi(self, window=6):
        """RSI指标

        计算公式：
        LC = REF(CLOSE, 1)
        RSI = SMA(MAX(CLOSE - LC, 0), N, 1) / SMA(ABS(CLOSE - LC), N, 1) * 100

        - 注意，这里如果要和股票软件/一般指标计算模组保持一致，则需要使用DMA，参数为1/window；
          如果要和原生计算公式保持一致，则使用SMA。
        # FIXME: DMA存在严重的计算偏差，具体表现为起点的不同会影响结果的不同
        """
        close = Factor('close')
        rsi = DMA(MAX(DELTA(close, 1), 0), 1 / window) / DMA(ABS(DELTA(close, 1)), 1 / window) * 100
        return rsi

    def rq_rsi3(self):
        return self.rq_rsi(window=3)

    def rq_rsi6(self):
        return self.rq_rsi(window=6)

    def rq_rsi12(self):
        return self.rq_rsi(window=12)

    def rq_rsi20(self):
        return self.rq_rsi(window=20)

    def rq_rsi40(self):
        return self.rq_rsi(window=40)

    def rq_rsi60(self):
        return self.rq_rsi(window=60)

    def rq_rsi120(self):
        return self.rq_rsi(window=120)

    def rq_rsi24(self):
        return self.rq_rsi(24)

    def _rq_rsimn(self, window1=6, window2=12):
        """两条RSI指标做差"""
        return self.rq_rsi(window1) - self.rq_rsi(window2)

    def rq_rsi612(self):
        return self._rq_rsimn(6, 12)

    def rq_rsi1224(self):
        return self._rq_rsimn(12, 24)

    def rq_rsi612_md(self, window=30):
        _rsi612 = self.rq_rsi612()
        return ABS(_rsi612 - MEDIAN(_rsi612, window))

    def rq_rsi1224_md(self, window=60):
        _rsi1224 = self.rq_rsi1224()
        return ABS(_rsi1224 - MEDIAN(_rsi1224, window))


class WilliamRate(MyRQFactors):

    def rq_wr(self, window=10):
        """威廉指数，原名叫“威廉超买超卖指数”（Williams Overbought/Oversold Index），简称WMS%R或%R，
        由拉瑞·威廉于1973年的《我如何赚取百万美元》一书中首先发表的，因而以他的名字命名。
        这个指标是一个振荡指标，是依股价的摆动点来度量股票/指数是否处于超买或超卖的现象。

        计算公式：
        WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
        """
        close = Factor('close')
        high = Factor('high')
        low = Factor('low')

        _hhv = HHV(high, window)
        _llv = LLV(low, window)
        wr = (_hhv - close) / (_hhv - _llv) * 100
        return wr

    def rq_lwr(self, window1=6, window2=10):
        """变异威廉指数，主要是使用快慢两根WR线与阈值作比较作为买入/卖出的判断
        1.低于20，可能超买见顶，可考虑卖出
        2.高于80，可能超卖见底，可考虑买进
        3.与RSI、MTM指标配合使用，效果更好

        计算公式
        WR1:100*(HHV(HIGH,N1)-CLOSE)/(HHV(HIGH,N1)-LLV(LOW,N1));
        WR2:100*(HHV(HIGH,N2)-CLOSE)/(HHV(HIGH,N2)-LLV(LOW,N2));
        """
        _wr1 = self.rq_wr(window1)
        _wr2 = self.rq_wr(window2)

        _v = _wr2 - _wr1
        return _v

    def rq_lwr_md(self, window=20):
        _lwr = self.rq_lwr()
        return ABS(_lwr - MEDIAN(_lwr, window))

    def rq_wvad(self, window=24, ratio=True):
        """威廉变异离散量（William's variable accumulation distribution），
        是一种将成交量加权的量价指标，用于测量从开盘价至收盘价期间，买卖双方各自爆发力的程度。

        - 20231031: 由于存在开盘即封板至全天收盘导致的high==low，在原公式基础上增加当日high是否等于low的判断，是则因子取值为1，否则输出原公式值。
                    这里取值为1的原因来自于因子本身含义的推导。(close-open)/(high-low)的含义是当日实体部分占波动幅度的大小，因此high==low表明
                    当日实体部分即整体波幅，等价于光头光脚大阳线或大阴线，所以取值为1.
        - TODOs：去量纲
        """
        factor_ = IF(EQUAL(Factor('high'), Factor('low')),
                     1, (Factor('close') - Factor('open')) / (Factor('high') - Factor('low')))
        if ratio:
            factor_ *= self.rq_turnover()
        else:
            factor_ *= Factor('volume')
        return SUM(factor_, window)

    def rq_wvad_md(self):
        _wvad = self.rq_wvad(30)
        return ABS(_wvad - MEDIAN(_wvad, 20))

    def rq_mawvad(self, window=20, ratio=True):
        return MA(self.rq_wvad(ratio=ratio), window)


class OBV(MyRQFactors):

    def rq_obv(self):
        """能量潮指标(On Balance Volume , OBV )

        计算方法：
        OBV=REF(OBV, 1) + sgn × VOLUME
        其中，sgn 是符号函数，其数值由下式决定：
        sgn=1 , CLOSE>REF(CLOSE, 1)
        sgn=0, CLOSE = REF(CLOSE, 1)
        sgn=-1 , CLOSE< REF(CLOSE, 1)
        这里的成交量是指成交股票的手数。

        - TODOs: 精细化
        """
        return Factor('OBV')

    def rq_obv6(self):
        return MA(self.rq_obv(), 6)

    def rq_obv20(self):
        return MA(self.rq_obv(), 20)

    # def rq_obv_zz500_abs(self, window=20, threshold=0.2):
    #     _zz500_mahv = self.rq_zz500_mahv()
    #     _obv_div20 = self._divmean(self.rq_obv(), window)
    #     _obv_div20_abs = ABS(_obv_div20)
    #     _obv_div20_zz500_abs = IF(_zz500_mahv <= threshold, _obv_div20_abs, _obv_div20)
    #     return _obv_div20_zz500_abs

    # def rq_obv20_dev(self):
    #     _obv = self.rq_obv()
    #     return (_obv - MA(_obv, 20)) / MA(_obv, 20)

    def rq_obv20_div_sum20(self):
        return self.rq_obv20() / SUM(Factor('volume'), 20)

    def rq_wobv(self, window=240):
        """去量纲的OBV指标

        """
        _delta_obv = DELTA(Factor('OBV'), window)
        _dm_delta_obv = _delta_obv / MA(Factor('volume'), 120)
        _f = EMA(_dm_delta_obv, window // 2)
        return _f

    def rq_wobv6(self):
        return self.rq_wobv(window=6)

    def rq_wobv20(self):
        return self.rq_wobv(window=20)

    def rq_wobv40(self):
        return self.rq_wobv(window=40)

    def rq_wobv60(self):
        return self.rq_wobv(window=60)

    def rq_wobv120(self):
        return self.rq_wobv(window=120)

    def rq_wobv240(self):
        return self.rq_wobv(window=240)

    def rq_wobv20_dm(self):
        """去量纲的OBV指标减去中值的幅度的绝对值
        - 效果不错
        """
        _f = self.rq_wobv(window=20)
        return ABS(_f - MEDIAN(_f, 240))

    def rq_wobv40_dm(self):
        """去量纲的OBV指标减去中值的幅度的绝对值
        - 效果不错
        """
        _f = self.rq_wobv(window=40)
        return ABS(_f - MEDIAN(_f, 240))

    def rq_wobv60_dm(self):
        """去量纲的OBV指标减去中值的幅度的绝对值
        - 效果不错
        """
        _f = self.rq_wobv(window=60)
        return ABS(_f - MEDIAN(_f, 240))

    def rq_wobv120_dm(self):
        """去量纲的OBV指标减去中值的幅度的绝对值
        - 效果不错
        """
        _f = self.rq_wobv(window=120)
        return ABS(_f - MEDIAN(_f, 240))

    def rq_wobv20_dmma(self):
        _f = self.rq_wobv()
        return MA(ABS(_f - MEDIAN(_f, 20)), 20)

    def rq_obv20_bband(self):
        _delta_obv = DELTA(self.rq_obv(), 240)
        _dm_delta_obv = _delta_obv / MA(Factor('volume'), 240)
        _f = EMA(_dm_delta_obv, 20)
        # _f_down, _f_mid, _f_up = self._rq_bband(_f, 60)
        return (_f - MA(_f, 60)) / (2 * STD(_f, 60))


class ATR(OBV):

    def _rq_tr(self, window=10):
        """N日真实振动幅度

        TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))),
                 ABS(LOW - REF(CLOSE, 1))), M1)
        """
        _high = Factor('high')
        _low = Factor('low')
        _close = Factor('close')
        _tr = SUM(MAX(MAX(_high - _low, ABS(_high - REF(_close, 1))),
                      ABS(_low, REF(_close, 1))), window)
        return _tr

    def _rq_atr(self, window):
        """N日均幅指标（Average TRUE Ranger）

        TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
        ATR = MA(TR, N)
        - 20231008：米筐含ATR和TR，其中ATR默认周期N==14，这里拆解为TR的计算
        """
        return MA(Factor('TR'), window)

    def rq_atr6(self):
        return self._rq_atr(6)

    def rq_atr14(self):
        return self._rq_atr(14)

    def rq_watr14(self):
        """14日平均真实波动除以移动120日平均减1"""
        _atr14 = self.rq_atr14()
        return _atr14 / MA(_atr14, 120) - 1

    def rq_watr20(self):
        _atr20 = self._rq_atr(20)
        return _atr20 / MA(_atr20, 120) - 1

    def rq_atr14_bband20(self):
        """ATR14的去量纲与股价波动率的差的EMA的绝对的log与去量纲的OBV相乘的结果
        - 效果分析：有一定的分组效果
        """
        _atr14 = self.rq_atr14()
        _watr14 = _atr14 / MA(_atr14, 20)
        _close_std = STD(Factor('close'), 20)
        diff_atr_bband = _watr14 * 1.5 - _close_std * 2

        _atr14_bband20 = LOG(ABS(EMA(diff_atr_bband, 20)))

        _wobv20 = self.rq_wobv()
        _v3 = _atr14_bband20 * _wobv20
        _v3_abs = ABS(_v3 - MEDIAN(_v3, 20))
        return _v3_abs


class MyFactors(WorldQuantAlpha101, Fundamentals, NPVI, RSRS, RSI, WilliamRate, ATR, Turnover):
    """因子集合与统一输出层

    """

    name = 'factors'


def get_rq_factor_dict(ftype='main'):
    if ftype == 'main':
        return MyRQFactors()
    elif ftype == 'base':
        return MyRQFactorsBase()
    elif ftype == 'financial':
        return Fundamentals()
    elif ftype == 'all':
        return MyFactors()
    else:
        raise ValueError("只支持一般因子和财务因子")


def is_quota_exceed_error(exception):
    str_exception = str(exception)
    quota_exceed = 'exceeds' in str_exception  # 包括login machine num exceeds / connection num exceeds
    conn_error = '10054' in str_exception
    if DEBUG:
        if quota_exceed:
            logger.warning("QuotaExceedError！Wait for 1 sec")
        elif conn_error:
            logger.warning("ConnectionResetError! Wait for 5 secs")
    return quota_exceed


@retry(stop_max_attempt_number=10, retry_on_exception=is_quota_exceed_error, wait_fixed=10)
def get_factor(factor_name, order_book_ids, start_date, end_date, factor_type=None, funcs=(), ts_kwargs={}, **kwargs):
    # 这里还是保留指定factor_type，减少列表存在性判断的时间
    # get_factor输入的处理财务因子，还有很多是量价因子，两者不在一个逻辑下处理，所以这里指定后区分处理模式
    if factor_type in ['base', 'deri']:
        factor_ = Fundamentals().get_financial_factor(factor_name)
    else:
        my_rq_factors = MyFactors()
        factor_ = my_rq_factors.get_factor(factor_name, funcs=funcs, ts_kwargs=ts_kwargs, **kwargs)  # my_rq_factors[factor_name]()
    if 'rsi' in factor_name:
        start_date = '2010-01-01'
    # 执行因子计算
    # 区分是否保留停牌NaN值点，默认保存
    class MyContext(ThreadingExecContext):
        def get_mask_for(self, order_book_id):
            return np.repeat(True, self.ndays)  # 返回一个全是True的mask
    if KEEPNAN:
        df = execute_factor(factor_, order_book_ids, start_date, end_date, exec_context_class=MyContext)
    else:
        df = execute_factor(factor_, order_book_ids, start_date, end_date)
    df.name = factor_name
    return df
    # return execute_factor(factor_, order_book_ids, start_date, end_date)


def pre_check_base_data(df: pd.DataFrame, factor_name, date_field, asset_field):
    """核查输入的数据是否标准

    核查内容包括：因子名是否重复、索引是否标准
    """
    _cols = df.columns
    _index_names = df.index.names
    # 索引检查
    if None in _index_names:
        assert date_field in _cols, f"输入数据不含时间列 {date_field}，请确认输入数据中包含列 {date_field} 或指定另一时间列给参数 date_field。"
        assert asset_field in _cols, f"输入数据不含资产列 {asset_field}，请确认输入数据中包含列 {asset_field} 或指定另一资产列给参数 asset_field。"
    else:
        assert date_field in _index_names, f"输入数据的索引不含时间列 {date_field}，请确认输入数据的索引包含列 {date_field} 或指定另一时间列索引给参数 date_field。"
        assert asset_field in _index_names, f"输入数据的索引不含资产列 {date_field}，请确认输入数据的索引包含列 {date_field} 或指定另一资产列索引给参数 asset_field。"

    # 因子重名检查
    if factor_name in _cols:
        overwrite = input(f"因子 {factor_name} 已存在，是否覆盖？[y/n]")
        if overwrite.lower() == 'y':
            logger.warning(f"即将覆盖因子 {factor_name} 并重新构建。")
            df.drop(columns=[factor_name], inplace=True)
        elif overwrite.lower() == 'n':
            logger.warning(f"即将跳过因子 {factor_name} 的构建。")
            return False
        else:
            raise ValueError("仅支持[y/n]的输入，重新输入！")
    return True


def get_cs_adjust_methods(with_underline=True):
    """横截面调整函数"""
    _methods = [
        '_csdivmean',
        '_cszscore',
        '_csdemean',
        '_csrank',
        '_csquantile',
        '_csscale',
        '_cstop', '_csbottom',
    ]
    if not with_underline:
        _methods = [method[1:] for method in _methods]
    return _methods


def cszscore(series: pd.Series):
    return (series - series.mean()) / series.std()


def csdivmean(series: pd.Series):
    return series / series.mean()


def csdemean(series: pd.Series):
    return series - series.mean()


def csrank(series: pd.Series, pct=False, ascending=True):
    return series.rank(pct=pct, ascending=ascending)


def csscale(series: pd.Series, scalar=1):
    return scalar * (series / series.sum())


def cstop(series: pd.Series, threshold=50, pct=True):
    """选取前threshold的股票"""
    rk = series.rank(pct=pct, ascending=True)  # 如果top含义采用大于计算，则ascending==True
    if pct:
        threshold /= 100
    ret = np.where(rk >= threshold, 1, 0)
    # numpy转换后需要再创建一个series，保持index和name不变
    return pd.Series(ret, index=rk.index, name=rk.name)


def csbottom(series: pd.Series, threshold=50, pct=True):
    rk = series.rank(pct=pct, ascending=True)
    if pct:
        threshold /= 100
    ret = np.where(rk <= threshold, 1, 0)
    # numpy转换后需要再创建一个series，保持index和name不变
    return pd.Series(ret, index=rk.index, name=rk.name)


def csquantile(series: pd.Series, q=10, reverse=True):
    """截面分组（默认0开始）"""
    qs = pd.qcut(series, q=q, labels=False, duplicates='drop')
    if reverse:
        # 当出现两组边界重复时，pd.qcut默认从0排序导致可能最大组不为9，所以此时需要整体向上挪N个单位
        if qs.nunique() != q:
            delta = q - 1 - qs.max()
            qs = qs + delta
    return qs


def cs_calculate(df, column_names, func_name, **kwargs):
    if '_' + func_name not in get_cs_adjust_methods():
        raise ValueError(f"不支持{func_name}操作，当前只支持：{get_cs_adjust_methods()}。")
    for col_name in column_names:
        new_col_name = f'{col_name}_{func_name}'
        df[new_col_name] = df.groupby('date')[col_name].apply(eval(func_name), **kwargs)
    return df


def split_funcs(func_list):
    """将输入的函数列表分割为时序和截面两块"""
    if len(func_list) == 0:
        return [], []
    cs_list = get_cs_adjust_methods(with_underline=False)
    idx_cs = []
    for i, func in enumerate(func_list):
        has_cs_func = False
        for cs_method in cs_list:
            if cs_method in func:
                idx_cs.append(i)
                has_cs_func = True
        if len(idx_cs) > 0 and not has_cs_func:
            break
    if len(idx_cs) > 0:
        ts_funcs = func_list[:min(idx_cs)]
        cs_funcs = func_list[min(idx_cs): max(idx_cs) + 1]
    else:
        ts_funcs = func_list
        cs_funcs = []
    return ts_funcs, cs_funcs


def add_factor(base_dataframe, factor_name, date_field='date', asset_field='order_book_id',
               factor_type=None, funcs=(), ts_kwargs={}, new_cs_funcs=(), new_cs_kwargs={}, **kwargs):
    """
    给定基本dataframe和因子名，在给定的dataframe上新增因子列

    Parameters
    ----------
    base_dataframe: 含有order_book_id和date（或类似性质的列）的长面板数据
    factor_name: 需要添加的因子名
    date_field: 日期列字段名称，默认'date'
    asset_field: 资产列字段名称，默认'order_book_id'
    factor_type: 财务因子的类型，默认基本财务因子
    funcs: 运算函数
    **kwargs : get_factor中传入的参数，主要是因子的参数，例如window等
    """
    # 检查输入数据是否标准
    _cont = pre_check_base_data(base_dataframe, factor_name, date_field, asset_field)
    if not _cont:
        return base_dataframe

    # 获取待计算因子资产和时间范围
    assets = base_dataframe[asset_field].unique().tolist()
    s_date = base_dataframe[date_field].min()
    e_date = base_dataframe[date_field].max()

    # TODO: 随着因子计算的需求迭代，后续将逐渐退出对米筐rqfactor的依赖，转为米筐数据+talib的本地处理
    #  难点1：部分时序回归的计算如何加速处理，例如DDNSR，DDNBT等。米筐rqfactor通过multiprocess实现加速，本地如何加速？
    #  难点2：如何保证使用计算的数据是有效的（这个保持从米筐读取数据的话是没问题的，米筐会做好pit的数据）

    # TODOd：计算过程还可以再优雅一点
    #  目前由于截面的操作_cs类必须要拿到get_factor外面来进行，因此大的框架必须是最后一个是截面处理
    #  中间的时序处理的叠加逻辑有两个：一种是还是在名称上通过尾缀的形式实现，另一种是改为参数输入式
    #  即新增一个参数processors保存一个或多个操作，get_factor内部只需要循环运行即可
    #  20231128：已实现纯·时序运算输入，下一步是支持截面算法输入
    ts_funcs, cs_funcs = split_funcs(funcs)
    df_factor = pd.DataFrame()
    # 针对字符串输入方式搜索横截面算法
    _cs_methods = get_cs_adjust_methods()
    for method in _cs_methods:
        if method in factor_name:
            base_factor_name, func_name = factor_name.split('_')
            MyFactors()._check_factor_exists(base_factor_name)
            factor_ = MyFactors().__getitem__(base_factor_name, **kwargs)
            df_factor = execute_factor(factor_, assets, s_date, e_date).stack().rename(factor_name)
            df_factor.index.names = [date_field, asset_field]
            logger.warning("即将返回基础因子<{}>的<{}>版本".format(base_factor_name, method))
            # 和基表merge后再进行界面操作，因为每个date的截面股票池不同
            df_all = base_dataframe.merge(df_factor.reset_index(), on=[date_field, asset_field], how='left')
            df_all[factor_name] = df_all.groupby(date_field)[factor_name].apply(eval(func_name))

    # 如果没有搜索到截面算法，则执行原因子调用程序
    if df_factor.empty:
        new_factor_name = '__'.join([factor_name] + ts_funcs)
        if len(ts_kwargs) > 0:
            new_factor_name += '__' + '__'.join([f"{k}_{v}" for k, v in ts_kwargs.items()])
        df_factor = get_factor(factor_name, assets, s_date, e_date,
                               factor_type=factor_type, funcs=ts_funcs, ts_kwargs=ts_kwargs, **kwargs).stack().rename(new_factor_name)
        df_factor.index.names = [date_field, asset_field]
        df_all = base_dataframe.merge(df_factor.reset_index(), on=[date_field, asset_field], how='left')
        # 针对列表输入方式继续搜索截面算法
        if len(cs_funcs) > 0:
            for cs_func in cs_funcs:
                logger.warning("对因子<{}>执行<{}>运算".format(new_factor_name, cs_func))
                df_all[new_factor_name] = df_all.groupby(date_field)[new_factor_name].apply(eval(cs_func))
                # 循环更新因子名称
                df_all.rename(columns={new_factor_name: new_factor_name + '__' + cs_func}, inplace=True)
                new_factor_name += '__' + cs_func
        if len(new_cs_funcs) == 1:
            # 目前abs操作只支持一个
            cs_func = new_cs_funcs[0]
            logger.warning("对因子<{}>执行<{}>运算，参数为<{}>".format(new_factor_name, cs_func, new_cs_kwargs))
            df_all[new_factor_name] = df_all.groupby(date_field)[new_factor_name].apply(eval(cs_func), **new_cs_kwargs)
            # 循环更新因子名称
            if len(new_cs_kwargs) > 0:
                df_all.rename(columns={new_factor_name: '__'.join([new_factor_name, cs_func, "__".join([f"{k}_{v}" for k, v in new_cs_kwargs.items()])])}, inplace=True)
                new_factor_name = '__'.join([new_factor_name, cs_func, "__".join([f"{k}_{v}" for k, v in new_cs_kwargs.items()])])
            else:
                df_all.rename(columns={new_factor_name: new_factor_name + '__' + cs_func}, inplace=True)
                new_factor_name += '__' + cs_func
    return df_all

def update_rqfactors_daily(factor_data_1d_path:str):
    if not os.path.exists(factor_data_1d_path):
        os.makedirs(factor_data_1d_path)
    base_factors = get_rq_factor_dict('base').get_factor_list(additional=False)
    all_factors = get_rq_factor_dict('all').get_factor_list(additional=False)
    # factors = base_factors + all_factors
    factors = []
    logger.info(f"开始更新米筐日度因子数据: 因子{len(factors)}个  {factors}")
    # factors.remove('excess_zz500_return')
    # factors.remove('sdk')
    # factors.remove('yield_curve')
    # factors.remove('zz500_excess_return')
    # factors.remove('zz500_pettm')
    # factors.remove('zz500_return')
    # factors.remove('ad')
    # factors.remove('aroon_osc')
    # factors.remove('atr14_bband20')
    # factors.remove('cfo_to_ev')
    # factors.remove('chosc')
    # factors.remove('coppock')
    # factors.remove('davol5')
    # factors.remove('ddnbt')
    # factors.remove('ddnbt120')
    # factors.remove('ddnbt240')
    # factors.remove('ddnbt60')
    # factors.remove('ddncr')
    # factors.remove('ddncr120')
    # factors.remove('ddncr240')
    # factors.remove('ddncr60')
    # factors.remove('ddnsr')
    # factors.remove('ddnsr120')
    # factors.remove('ddnsr240')
    # factors.remove('ddnsr60')
    # factors.remove('excess_zz500_return')
    # factors.remove('fsl')
    # factors.remove('jdqs')
    # factors.remove('nvi20_deri')
    # factors.remove('nvid240_z504')
    # factors.remove('obv20_bband')
    # factors.remove('pvi20_deri')
    # factors.remove('pvid240_z504')
    # factors.remove('pvid252_z504')
    # factors.remove('rsrs10_zz500_abs')
    # factors.remove('rsrs20_zz500_abs')
    # factors.remove('rsrs40_zz500_abs')
    # factors.remove('rsrs_zz500_abs')
    # factors.remove('sales_service_cash_to_or')
    # factors.remove('sdk')
    # factors.remove('sws')
    # factors.remove('tobt')
    # factors.remove('tv20_div_tv120')
    # factors.remove('tv3_div_tv120')
    # factors.remove('tv5_div_tv120')
    # factors.remove('tv_div_tv120')
    # factors.remove('tvrevs10')
    # factors.remove('tvrevs10_demean')
    # factors.remove('tvrevs20')
    # factors.remove('tvrevs20_demean')
    # factors.remove('tvrevs20_divmean')
    # factors.remove('tvrevs5')
    # factors.remove('tvroc12')
    # factors.remove('tvroc20')
    # factors.remove('tvroc24')
    # factors.remove('tvroc6')
    # factors.remove('uupbt')
    # factors.remove('uupbt120')
    # factors.remove('uupbt240')
    # factors.remove('uupbt60')
    # factors.remove('uupcr')
    # factors.remove('uupcr120')
    # factors.remove('uupcr240')
    # factors.remove('uupcr60')
    # factors.remove('uupsr')
    # factors.remove('uupsr120')
    # factors.remove('uupsr240')
    # factors.remove('uupsr60')
    # factors.remove('vol120')
    # factors.remove('vroc12')
    # factors.remove('vroc20')
    # factors.remove('vroc24')
    # factors.remove('vroc6')
    # factors.remove('watr14')
    # factors.remove('watr20')
    # factors.remove('wobv')
    # factors.remove('wobv120')
    # factors.remove('wobv120_dm')
    # factors.remove('wobv20')
    # factors.remove('wobv20_dm')
    # factors.remove('wobv20_dmma')
    # factors.remove('wobv240')
    # factors.remove('wobv40')
    # factors.remove('wobv40_dm')
    # factors.remove('wobv6')
    # factors.remove('wobv60')
    # factors.remove('wobv60_dm')
    # factors.remove('yield_curve')
    # factors.remove('zz500_excess_return')
    # factors.remove('zz500_hv')
    # factors.remove('zz500_mahv')
    # factors.remove('zz500_pettm')
    # factors.remove('zz500_return')
    # 增加入库米筐因子
    factors.append('market_cap_3')
    factors.append('total_turnover')
        
    logger.info(f"启动米筐日度因子数据更新 因子数量：{len(factors)}")
    
    df_basic = rqdatac.all_instruments(type='CS', market='cn', date=None)
    order_book_ids = df_basic['order_book_id'].unique().tolist()

    datas = list()
    for factor_name in factors:
        logger.info(f"启动{factor_name}因子数据更新")
        # 确认本地文件路径
        fp_factor = os.path.join(factor_data_1d_path, f"{factor_name}.feather")
        if os.path.exists(fp_factor):
            df_local_factor = pd.read_feather(fp_factor)
            df_local_factor.set_index(['date'], inplace=True)
            local_max_date = df_local_factor.index.max()
            update_start_date = rqdatac.get_next_trading_date(local_max_date)
        else:
            # 无文件，首次生成，直接设置默认起点
            df_local_factor = pd.DataFrame()
            update_start_date = pd.to_datetime('2001-01-01')
            logger.warning(f"{factor_name}因子尚未生成过，{fp_factor}将从默认起点:{update_start_date}开始生成数据，耗时较久")
            
        now = datetime.now()
        if now.hour >= 16:
            update_end_date = rqdatac.get_latest_trading_date()
        else:
            update_end_date = rqdatac.get_previous_trading_date(now)
        if update_start_date > update_end_date:
            logger.info(f"{factor_name}已是最新数据, 跳过 时间范围:{update_start_date} to {update_end_date}")
            continue
        # 仅保留新数据部分
        logger.info(f"正在下载{factor_name}因子数据， 时间范围：{update_start_date} to {update_end_date}")
        # 定义因子参数
        if factor_name == "market_cap_3":
            ts_funcs = [("ma", {"window":20}),]
            df_new_factor = get_factor(factor_name, order_book_ids, update_start_date, update_end_date, ts_funcs=ts_funcs)
        else:
            df_new_factor = get_factor(factor_name, order_book_ids, update_start_date, update_end_date)
        # print(df_new_factor)
        update_end_date = df_new_factor.index.max()
        df_all_factor_data = pd.concat([df_local_factor, df_new_factor]).sort_index()
        df_all_factor_data.index.name = 'date'
        df_all_factor_data.reset_index(inplace=True)
        start_time = time.time()
        df_all_factor_data.to_feather(fp_factor)
        end_time = time.time()
        logger.info(f"{factor_name}因子更新完成 {fp_factor}写入完成, 耗时：{round(end_time-start_time, 2)}s {round(len(df_all_factor_data)/(end_time-start_time), 2)}row/s")
        datas.append((fp_factor, factor_name, df_new_factor))
    return datas


# base_factors = get_rq_factor_dict('base').get_factor_list(additional=False)
# all_factors = get_rq_factor_dict('all').get_factor_list(additional=False)
# logger.info("目前支持自建总因子数量：%s。因子列表如下：\n"
#             "{}" % len(all_factors), all_factors)
# logger.info("其中基础/中间因子数量：%s。因子列表如下：\n"
#             "{}" % len(base_factors), base_factors)
# ts_methods = get_rq_factor_dict('base').get_adjust_methods()
# logger.info("当前支持的个股时序调整方法数量：%s。方法列表如下，详情见函数文档：\n"
#             "{}" % len(ts_methods), ts_methods)
# cs_methods = get_cs_adjust_methods()
# logger.info("当前支持的横截面调整方法数量：%s。方法列表如下，详情见函数文档：\n"
#             "{}" % len(cs_methods), cs_methods)

__all__ = ['update_rqfactors_daily']

def abs_to_quantile(x, q=0.5, rank=False):
    """
    Calculates the absolute distance from the quantile value.
    Parameters
    ----------
    x: the time series to calculate the feature of
    q: Quantile to compute, which must be between 0 and 1 inclusive.
    rank: Whether compute quantiles based on ranks, default False

    Returns
    -------
    the value of this feature, type list
    """

    if rank:
        # 找出非 NaN 元素
        non_nan_arr = x[~np.isnan(x)]

        # 对非 NaN 元素使用 rankdata 获取排名
        ranks = stats.rankdata(non_nan_arr, method='ordinal')
        reversed_ranks = x.size - ranks + 1
        # 创建一个和原始数组同样大小的数组来存储排名，初始值设为 NaN
        final_ranks = np.full(x.shape, np.NaN)

        # 将排名放回非 NaN 位置
        final_ranks[~np.isnan(x)] = reversed_ranks
        x = pd.Series(final_ranks, index=x.index)
    q_val = np.nanquantile(x, q)
    x_diff = x - q_val
    return np.abs(x_diff)


def abs_to_mean(x):
    return np.abs(x - np.nanmean(x))


def decode_tsfresh_columns_name(column_name):
    """给定TSFresh返回的列名解构TSFresh的参数"""

    # 其次判断是否有abs函数
    abs_func = None
    abs_func_params = {}
    abs_funcs = ['abs_to_quantile', 'abs_to_mean']
    for each_abs_func in abs_funcs:
        if each_abs_func in column_name:
            spliter = '__' + each_abs_func + '__'
            column_name, abs_func_params_str = column_name.split(spliter)
            if len(abs_func_params_str) > 0:
                for param in abs_func_params_str.split("__"):
                    k, v = param.rsplit("_", 1)
                    try:
                        abs_func_params[k] = eval(v)
                    except NameError:
                        abs_func_params[k] = v
            abs_func = each_abs_func
            break

    # 首先判断结尾是否多余，针对无参数的tsfresh或自定义abs_to函数
    # if column_name.endswith('__'):
    #     column_name = column_name[:-2]
    if column_name.endswith('__'):
        column_name = column_name[:-2]
    if "__" not in column_name:
        return column_name, None, None, abs_func, abs_func_params
    elif len(column_name.split('__')) == 2:
        factor_name, tsfresh_func = column_name.split('__')
        return factor_name, tsfresh_func, {}, abs_func, abs_func_params
    factor_name, tsfresh_func, params_str = column_name.split('__', 2)
    params = {}
    for param in params_str.split("__"):
        # 存在
        k, v = param.rsplit("_", 1)
        try:
            # if '"' in v:
            #     params[k] = eval(v)
            # else:
            # params[k] = int(v)
            if len(re.findall('[0-9]+', v)) > 0:
                params[k] = eval(v)
            else:
                params[k] = v
        except NameError:
            params[k] = v
    return factor_name, tsfresh_func, params, abs_func, abs_func_params


def decode_window(tsfunc_window):
    """更新后的tsfunc会带有旧window（默认最后的数字）"""
    if not tsfunc_window:
        return None, None
    results = re.findall('([^0-9]+)([0-9]+)', tsfunc_window)
    if len(results) > 0:
        return results[0]
    else:
        raise ValueError(f"无法解析：{tsfunc_window}")
