"""
产品净值计算脚本 - 优化版（无pathlib）
根据原始持仓、资产等数据，计算产品净值、收益、风险指标等。
"""

import logging
import os
import shutil
import sys
import warnings
from typing import Dict, Optional, Tuple, List, Any
from glob import glob
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import rqdatac
from tabulate import tabulate
from pathlib import Path
# 自动把 D:\code 加入 Python 搜索路径
sys.path.append(str(Path(__file__).parent.parent))
#
# rqdatac.init(username="license",
#              password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=",
#              use_pool=True, max_pool_size=8)

# 导入邮件管理器（原项目模块）
from ultis.email_manager import EmailManager
from ultis.rq_date_request import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ==================== 常量与配置 ====================
class MarketConstants:
    """市场相关常量"""
    BENCHMARK_CODES = ['000300.XSHG', '000905.XSHG', '000510.XSHG', '000852.XSHG']
    BENCHMARK_NAMES = ['hs300', 'zz500', 'a500', 'zz1000']
    FUTURES_ALLOWED_PREFIXES = ('IH', 'IF', 'IC', 'IM')
    FUTURES_ALLOWED_EXCHANGES = ('CFFEX', 'DCE')
    DEFAULT_MARGIN_RATIO = 0.12


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config = None

    def load(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        return self._config

    def get_date(self) -> str:
        date = self._config.get('date', 'current')
        if date == 'current':
            date = datetime.now().strftime('%Y%m%d')
        return date


# ==================== 数据获取层 ====================
class MarketDataFetcher:
    """市场数据获取器（封装rqdatac调用）"""

    def __init__(self, date: str, pre_date: str):
        self.date = date
        self.pre_date = pre_date

    @staticmethod
    def get_previous_trading_date(date: str) -> str:
        try:
            return rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
        except Exception as e:
            logger.error(f"获取前一个交易日失败: {e}")
            return date

    @staticmethod
    def get_trading_dates(start_date: str, end_date: str) -> str:
        try:
            rk_dates = rqdatac.get_trading_dates(start_date=start_date, end_date=end_date)
            rk_dates = [str(_date).replace('-', '') for _date in rk_dates]
            return rk_dates
        except Exception as e:
            logger.error(f"获取前一个交易日失败: {e}")
            return []

    def get_benchmark_info(self) -> pd.DataFrame:
        """获取基准指数数据（沪深300、中证500、中证A500、中证1000）"""
        codes = MarketConstants.BENCHMARK_CODES
        benchmarks = MarketConstants.BENCHMARK_NAMES
        try:
            df = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date,
                frequency='1d', fields=None, adjust_type='pre',
                skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
        except Exception:
            logger.warning(f"获取指数日线数据失败 ({self.date})，使用实时行情")
            prev_df = rqdatac.get_price(
                codes, start_date=self.pre_date, end_date=self.pre_date,
                frequency='1d', fields=None, adjust_type='pre',
                skip_suspended=False, market='cn', expect_df=True
            ).reset_index()[['order_book_id', 'close']].rename(columns={'close': 'prev_close'})
            cur_df = rqdatac.current_minute(codes, skip_suspended=False).reset_index()[['order_book_id', 'close']]
            df = pd.merge(prev_df, cur_df, on='order_book_id')

        df = df.rename(columns={'order_book_id': 'code'})
        df['chg'] = df['close'] / df['prev_close'] - 1
        df['benchmark'] = df['code'].apply(lambda x: benchmarks[codes.index(x)])
        return df.set_index('benchmark')[['close', 'prev_close', 'chg']]

    def get_futures_info(self) -> pd.DataFrame:
        """获取期货合约信息（收盘价、结算价、乘数等）"""
        fut_mkt = rqdatac.all_instruments(type='Future', market='cn', date=self.date)
        # 筛选交易所和品种
        mask = (fut_mkt['order_book_id'].str.startswith(MarketConstants.FUTURES_ALLOWED_PREFIXES) &
                fut_mkt['exchange'].isin(MarketConstants.FUTURES_ALLOWED_EXCHANGES) &
                (~fut_mkt['symbol'].str.contains('连续') | fut_mkt['symbol'].str.contains('国债')))
        fut_mkt = fut_mkt[mask].reset_index(drop=True)
        codes = fut_mkt['order_book_id'].tolist()
        fut_df = fut_mkt[['order_book_id', 'exchange', 'contract_multiplier']].rename(columns={'order_book_id': 'code'})

        try:
            price_df = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date, frequency='1d',
                fields=None, adjust_type='pre', skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
            price_df = price_df[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']]
            price_df = price_df.rename(
                columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
            fut_df = pd.merge(fut_df, price_df, on='code', how='left')
        except Exception as e:
            logger.warning(f"获取期货日线数据失败 ({self.date})，尝试使用分钟线: {e}")
            fut_df = self._fill_futures_with_minute_data(fut_df, codes)
        return fut_df.set_index('code')

    def _fill_futures_with_minute_data(self, fut_df: pd.DataFrame, codes: List[str], paths: str=None) -> pd.DataFrame:
        """使用分钟线数据补充期货信息（原有逻辑完整保留）"""
        # 获取前一日日线数据
        prev_df = rqdatac.get_price(
            codes, start_date=self.pre_date, end_date=self.pre_date,
            frequency='1d', fields=None, adjust_type='pre',
            skip_suspended=False, market='cn', expect_df=True
        ).reset_index()
        prev_df = prev_df[['order_book_id', 'close', 'settlement']].rename(
            columns={'order_book_id': 'code', 'close': 'preclose', 'settlement': 'presettlement'}
        )
        fut_df = pd.merge(fut_df, prev_df, on='code', how='left')

        # 读取盘中分钟线数据（假设文件存在）
        # E:\code\generate_split_system\data\raw\real_minute
        # minute_file = f'/home/zhanggh/testscripts/test_data/{self.date}_data.csv'
        minute_file = rf'E:\code\generate_split_system\data\raw\real_minute\{self.date}_data.csv'
        # minute_file = paths
        if not os.path.exists(minute_file):
            logger.error(f"分钟线数据文件不存在: {minute_file}")
            return fut_df

        minute_df = pd.read_csv(minute_file, index_col=0).rename(columns={'order_book_id': 'code'})
        minute_df = minute_df.drop_duplicates(subset=None, keep='last', ignore_index=True)
        minute_df['datetime'] = minute_df['datetime'].astype(str)
        minute_df[['date', 'time']] = minute_df['datetime'].str.split(' ', expand=True)
        minute_df['date'] = minute_df['date'].str.replace('-', '')
        minute_df['time'] = minute_df['time'].str.replace(':', '')

        # 收盘价：取15:00的close
        close_df = minute_df[minute_df['time'] == '150000'].reset_index()
        if not close_df.empty:
            fut_df = pd.merge(fut_df, close_df[['code', 'close']], on='code', how='left')

        # 结算价：取14:00-15:00成交量加权均价
        settle_df = minute_df[(minute_df['time'] > '140000') & (minute_df['time'] <= '150000')]
        grouped = settle_df.groupby('code')[['volume', 'total_turnover']].sum().reset_index()
        grouped['settlement'] = grouped['total_turnover'] / grouped['volume']
        fut_df = pd.merge(fut_df, grouped[['code', 'settlement']], on='code', how='left')
        if 'contract_multiplier' in fut_df.columns:
            fut_df['settlement'] = fut_df['settlement'] / fut_df['contract_multiplier']  # 转换为单位价格
        return fut_df

    def get_stock_price_info(self, time_periods: List[Tuple[str, str]]) -> pd.DataFrame:
        """获取股票价格信息（昨收、今收、全天VWAP、各时段VWAP）"""
        all_stk = rqdatac.all_instruments(type='CS', market='cn', date=self.date)
        codes = all_stk['order_book_id'].unique().tolist()
        try:

            predaily = rqdatac.get_price(
                codes, start_date=self.pre_date, end_date=self.pre_date, frequency='1d',
                fields=['close'], adjust_type='pre', skip_suspended=False, market='cn',
                expect_df=True).reset_index().rename(columns={'close': 'prev_close'})
            daily = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date, frequency='1d',
                fields=['close', 'total_turnover', 'volume'],
                adjust_type='pre', skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
            daily = pd.merge(daily, predaily[['order_book_id', 'prev_close']], on='order_book_id', how='left')
            daily['allday_vwap'] = np.where(daily['volume'] > 0, daily['total_turnover'] / daily['volume'],
                                            daily['close'])
            daily['date'] = daily['date'].dt.strftime('%Y%m%d')
            daily = daily[['order_book_id', 'date', 'prev_close', 'close', 'allday_vwap']]

            minute = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date, frequency='1m',
                fields=None, adjust_type='pre', skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
            minute['date'] = minute['datetime'].dt.strftime('%Y%m%d')
            minute['time'] = minute['datetime'].dt.strftime('%H%M%S')

            for st, et in time_periods:
                label = f'{st}-{et}_vwap'
                period_data = minute[minute['time'].between(st, et, inclusive='right')]
                grouped = period_data.groupby(['order_book_id', 'date'])[
                    ['total_turnover', 'volume']].sum().reset_index()
                grouped[label] = np.where(grouped['volume'] > 0, grouped['total_turnover'] / grouped['volume'], 0)
                daily = pd.merge(daily, grouped[['order_book_id', 'date', label]], on=['order_book_id', 'date'],
                                 how='left')
        except Exception as e:
            logger.warning(f"获取股票数据失败 ({self.date})，使用简化数据: {e}")
            daily = self._get_simple_stock_price(codes)
        daily = daily.rename(columns={'order_book_id': 'code', 'prev_close': 'preclose'})
        daily['code'] = daily['code'].apply(lambda x: int(str(x)[:6]))
        daily['date'] = daily['date'].astype(int)
        return daily.set_index('code')

    def _get_simple_stock_price(self, codes: List[str]) -> pd.DataFrame:
        """降级方案：仅获取收盘价和前收盘"""
        today = rqdatac.current_minute(codes, fields=['close'], skip_suspended=False).reset_index()
        yesterday = rqdatac.get_price(
            codes, start_date=self.pre_date, end_date=self.pre_date, frequency='1d',
            fields=['close'], adjust_type='pre', skip_suspended=False, market='cn', expect_df=True
        ).reset_index().rename(columns={'close': 'prev_close'})
        df = pd.merge(today[['order_book_id', 'close']], yesterday[['order_book_id', 'prev_close']], on='order_book_id')
        df['date'] = self.date
        df['allday_vwap'] = 0.0
        return df


# ==================== 产品净值计算核心 ====================
class ProductNetCalculator:
    """产品净值计算器（优化版）"""

    def __init__(self, date: str, config: Dict):
        self.date = date
        self.pre_date = MarketDataFetcher.get_previous_trading_date(date)
        self.product_info = config.get('product_info', {})
        self.margin_ratio = config.get('margin_ratio', MarketConstants.DEFAULT_MARGIN_RATIO)

        # 远程路径
        self.data_path = config.get('data_path')
        self.standard_path = os.path.join(self.data_path, 'data', 'standarddata')
        self.out_path = os.path.join(self.data_path, 'data', 'out')
        self.mo_order_path = os.path.join(self.data_path, 'data', 'mo_order')
        self.dw_path = os.path.join(self.data_path, 'data')
        self.net_email_path = os.path.join(self.data_path, 'data', 'raw', 'net_email')

        # 本地路径（读取时优先使用，不存在时退回到远程）
        self.local_path = config.get('local_path')
        self.local_standard_path = os.path.join(self.local_path, 'data', 'standarddata')
        self.local_out_path = os.path.join(self.local_path, 'data', 'out')
        self.local_mo_order_path = os.path.join(self.local_path, 'data', 'mo_order')
        self.local_dw_path = os.path.join(self.local_path, 'data')
        self.local_net_email_path = os.path.join(self.local_path, 'data', 'raw', 'net_email')

        self._download_emails()

        # 初始化市场数据获取器
        self.market = MarketDataFetcher(self.date, self.pre_date)
        self.index_df = self.market.get_benchmark_info()
        self.fut_info = self.market.get_futures_info()
        self.adj_periods = self._get_adj_periods()
        self.stk_price = self.market.get_stock_price_info(self.adj_periods)

    # ================================================================
    # 文件路径辅助：优先本地，不存在则使用远程
    # ================================================================
    def _resolve_read_path(self, local_path, remote_path):
        """读取文件时优先使用本地路径，不存在则退回到远程路径"""
        if os.path.exists(local_path):
            logger.debug(f'  读取本地文件: {local_path}')
            return local_path
        logger.debug(f'  本地文件不存在，读取远程: {remote_path}')
        return remote_path

    def _get_adj_periods(self) -> List[Tuple[str, str]]:
        """获取调仓时段（从配置中提取）"""
        periods = []
        for product_info in self.product_info.values():
            acct_info_dict = product_info.get('acct_info', {})
            for acct_info in acct_info_dict.values():
                start = acct_info.get('adj_algo_starttime')
                end = acct_info.get('adj_algo_endtime')
                if start and end:
                    periods.append((start, end))
        return list(set(periods))

    def _download_emails(self):
        """下载邮件附件中的净值数据"""
        product_chinese_names = [info.get('product_chinese_name') for info in self.product_info.values()]
        format_predate = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'
        manager = EmailManager()
        for name in product_chinese_names:
            if name == '配邦中圣1号':
                name = '配邦中圣'
            match_file = self._match_file(self.net_email_path, name, format_predate)
            if not match_file or not os.path.exists(match_file):
                match_file = self._match_file(self.local_net_email_path, name, format_predate)
            if not match_file or not os.path.exists(match_file):
                try:
                    manager.download_attachments_by_keyword([name, format_predate], save_dir=self.net_email_path,
                                                            file_extensions=['.xlsx'])
                except Exception as e:
                    logger.warning(f"下载 {self.pre_date} {name} 产品净值邮件出错: {e}")
        manager.logout()

        # 远程已下载完成，同步到本地
        os.makedirs(self.local_net_email_path, exist_ok=True)
        for fn in os.listdir(self.net_email_path):
            src_file = os.path.join(self.net_email_path, fn)
            dst_file = os.path.join(self.local_net_email_path, fn)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                logger.debug(f'  同步邮件附件到本地: {fn}')

    @staticmethod
    def _match_file(path: str, key: str, date_str: str) -> Optional[str]:
        """匹配包含日期和关键字的文件"""
        pattern = os.path.join(path, f"*{date_str}*")
        for file in glob(pattern):
            if key in file:
                return file
        return None

    def _read_csv(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        """读取CSV文件，如果不存在则返回None"""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                logger.error(f"读取文件失败 {path}: {e}")
        return None

    def _read_product_files(self, product: str, subdir: str, suffix: str = '', date: str = None) -> Optional[
        pd.DataFrame]:
        """统一读取产品相关文件（支持多文件合并）"""
        if date is None:
            date = self.date
        # 优先使用本地路径
        local_base = os.path.join(self.local_standard_path, subdir, date)
        remote_base = os.path.join(self.standard_path, subdir, date)
        base_dir = self._resolve_read_path(local_base, remote_base)
        if subdir in ['assets', 'pos']:
            pattern = os.path.join(base_dir, f"{date}*{product}{suffix}.csv")
            files = glob(pattern)
            if not files:
                return None
            dfs = []
            for f in files:
                df = self._read_csv(f)
                if df is not None:
                    dfs.append(df)
            return pd.concat(dfs, ignore_index=True) if dfs else None
        else:
            file_path = os.path.join(base_dir, f"{date}_{product}{suffix}.csv")
            return self._read_csv(file_path)

    # -------------------- 期货相关计算 --------------------
    def _get_futures_assets(self, product: str) -> Tuple[float, float, float]:
        """获取期货静态权益、出金、入金"""
        fut_assets = self._read_product_files(product, 'fut', '_fut_assets')
        if fut_assets is None:
            return 0.0, 0.0, 0.0
        return fut_assets['DynamicRights'].sum(), fut_assets['Withdraw'].sum(), fut_assets['Deposit'].sum()

    def _get_options_assets(self, product: str) -> Tuple[float, float]:
        """获取期权总资产和当日盈亏"""
        opt_assets = self._read_product_files(product, 'opt_data', '_opt_assets')
        if opt_assets is None:
            return 0.0, 0.0
        total = opt_assets['总资产'].sum()
        pnl = opt_assets['估算浮盈'].sum() if '估算浮盈' in opt_assets else 0.0
        return total, pnl

    def _get_deposits_withdrawals(self, product: str) -> float:
        """获取出入金总额"""
        dw_path = os.path.join(self.dw_path, 'deposits_withdrawals.csv')
        dw_path_local = dw_path.replace(self.dw_path, self.local_dw_path, 1)
        dw_path = dw_path_local if os.path.exists(dw_path_local) else dw_path
        if not os.path.exists(dw_path):
            return 0.0
        dw_df = pd.read_csv(dw_path, index_col=0)
        mask = (dw_df['date'].astype(str) == self.date) & (dw_df['product'] == product)
        return dw_df.loc[mask, 'deposite_withdrawwals'].sum() if mask.any() else 0.0

    def _get_previous_net(self, product: str) -> Tuple[float, float, float]:
        """获取前一日净资产（收盘价、结算价）及期货静态权益"""
        pre_file = os.path.join(self.out_path, 'net_data', self.pre_date,
                                f'{self.pre_date}_{product}_net_info.csv')
        pre_file_local = pre_file.replace(self.out_path, self.local_out_path, 1)
        pre_file = pre_file_local if os.path.exists(pre_file_local) else pre_file
        if not os.path.exists(pre_file):
            return 0.0, 0.0, 0.0
        pre_df = pd.read_csv(pre_file)
        if pre_df.empty:
            return 0.0, 0.0, 0.0
        return pre_df.loc[0, '产品净资产(收盘价)'], pre_df.loc[0, '产品净资产(结算价)'], pre_df.loc[0, '期货户静态权益']

    def _get_pre_net_from_mail(self, product_chinese: str) -> float:
        """从邮件附件中获取前一日单位净值"""
        format_predate = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'
        if product_chinese == '配邦中圣1号':
            product_chinese = '配邦中圣'
        file_path = self._match_file(self.net_email_path, product_chinese, format_predate)
        if not file_path or not os.path.exists(file_path):
            file_path = self._match_file(self.local_net_email_path, product_chinese, format_predate)
        if not file_path:
            return 0.0
        data = pd.read_excel(file_path)[:1].rename(columns={'日期': '净值日期', '资产份额净值(元)': '单位净值'})
        pre_net = data.loc[data['净值日期'] == format_predate, '单位净值']
        return pre_net.iloc[0] if not pre_net.empty else 0.0

    def _calc_futures_diff_pnl(self, product: str) -> Tuple[float, float, float]:
        """计算期货结算价与收盘价的差异盈亏、持仓市值和保证金"""
        fut_pos = self._read_product_files(product, 'fut', '_fut_pos')
        if fut_pos is None or fut_pos.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        fut_pos = fut_pos[fut_pos['code'].isin(self.fut_info.index)]
        if fut_pos.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        fut_pos = fut_pos.merge(
            self.fut_info[['close', 'settlement', 'presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )
        fut_pos['diff_pnl'] = (fut_pos['settlement'] - fut_pos['close']) * fut_pos['Direction'] * fut_pos['vol'] * \
                              fut_pos['contract_multiplier']
        fut_pos['amount'] = fut_pos['Direction'] * fut_pos['vol'] * fut_pos['close'] * fut_pos['contract_multiplier']
        fut_pos['settlement_amount'] = fut_pos['contract_multiplier'] * fut_pos['settlement'] * fut_pos['Direction'] * \
                                       fut_pos['vol']

        # 按用户分组
        quant = fut_pos[fut_pos['user'] == 'quant'] if 'user' in fut_pos else pd.DataFrame()
        cta = fut_pos[fut_pos['user'] == 'cta'] if 'user' in fut_pos else pd.DataFrame()

        quant_diff = quant['diff_pnl'].sum() if not quant.empty else 0.0
        quant_hold = quant['amount'].sum() if not quant.empty else 0.0
        quant_margin = abs(quant['settlement_amount'].sum()) * self.margin_ratio if not quant.empty else 0.0

        cta_diff = cta['diff_pnl'].sum() if not cta.empty else 0.0
        cta_hold = cta['amount'].sum() if not cta.empty else 0.0
        cta_margin = 0.0
        if not cta.empty:
            margin_tmp = cta[['code', 'settlement_amount']].copy()
            # print(margin_tmp)
            margin_tmp['abs_amount'] = margin_tmp['settlement_amount'].abs()
            margin_tmp = margin_tmp.groupby('code').agg({'abs_amount': 'sum', 'settlement_amount': 'sum'}).reset_index()
            margin_tmp['settlement_amount'] = np.where(margin_tmp['settlement_amount'] == 0,
                                                       margin_tmp['abs_amount'] * 0.5, margin_tmp['settlement_amount'])
            # print(margin_tmp)
            cta_margin = margin_tmp['settlement_amount'].abs().sum() * self.margin_ratio
        # print(product)
        print(f'product: {product}, cta_diff:{cta_diff}, quant_diff:{quant_diff}, fut_diff:{cta_diff + quant_diff}')
        if product == 'PBTZ2H':  # 特殊产品只取CTA
            return quant_diff, cta_diff, quant_hold, cta_hold, cta_margin
        else:
            return quant_diff, cta_diff, quant_hold, quant_hold + cta_hold, quant_margin + cta_margin

    def _calc_futures_order_pnl(self, product: str) -> Tuple[float, float, float, float]:
        """计算期货交易盈亏（quant/cta）"""
        fut_order = self._read_product_files(product, 'fut', '_fut_order')
        if fut_order is None or fut_order.empty:
            return 0.0, 0.0, 0.0, 0.0
        fut_order = fut_order[fut_order['code'].isin(self.fut_info.index)]
        if fut_order.empty:
            return 0.0, 0.0, 0.0, 0.0
        fut_order = fut_order.merge(
            self.fut_info[['presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )
        fut_order['order_pnl'] = fut_order['Direction'] * fut_order['filled_vol'] * fut_order['contract_multiplier'] * (
                    fut_order['presettlement'] - fut_order['filled_price'])

        quant = fut_order[fut_order['user'] == 'quant']
        cta = fut_order[fut_order['user'] == 'cta']
        quant_pnl = quant['order_pnl'].sum() if not quant.empty else 0.0
        quant_fee = quant['UsedFee'].sum() if not quant.empty else 0.0
        cta_pnl = cta['order_pnl'].sum() if not cta.empty else 0.0
        cta_fee = cta['UsedFee'].sum() if not cta.empty else 0.0
        return quant_pnl, quant_fee, cta_pnl, cta_fee

    def _calc_futures_pos_pnl(self, product: str) -> Tuple[float, float, float, float, float, float]:
        """计算期货持仓盈亏（量化/CTA）"""
        fut_pos = self._read_product_files(product, 'fut', '_fut_pos')
        if fut_pos is None or fut_pos.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        fut_pos = fut_pos[fut_pos['code'].isin(self.fut_info.index)]
        if fut_pos.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        fut_pos = fut_pos.merge(
            self.fut_info[['close', 'settlement', 'presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )
        quant = fut_pos[fut_pos['user'] == 'quant']
        cta = fut_pos[fut_pos['user'] == 'cta']

        quant_hold_val = quant['Direction'] * quant['vol'] * quant['contract_multiplier'] * quant['settlement']
        quant_hold_val = quant_hold_val.sum() if not quant.empty else 0.0
        quant_hold_pnl = ((quant['settlement'] - quant['presettlement']) * quant['vol'] * quant['contract_multiplier'] *
                          quant['Direction']).sum() if not quant.empty else 0.0
        quant_diff_pnl = ((quant['settlement'] - quant['close']) * quant['Direction'] * quant['vol'] * quant[
            'contract_multiplier']).sum() if not quant.empty else 0.0

        cta_hold_val = cta['Direction'] * cta['vol'] * cta['contract_multiplier'] * cta['settlement']
        cta_hold_val = cta_hold_val.sum() if not cta.empty else 0.0
        cta_hold_pnl = ((cta['settlement'] - cta['presettlement']) * cta['vol'] * cta['contract_multiplier'] * cta[
            'Direction']).sum() if not cta.empty else 0.0
        cta_diff_pnl = ((cta['settlement'] - cta['close']) * cta['Direction'] * cta['vol'] * cta[
            'contract_multiplier']).sum() if not cta.empty else 0.0

        return quant_hold_val, quant_hold_pnl, quant_diff_pnl, cta_hold_val, cta_hold_pnl, cta_diff_pnl

    # -------------------- 股票相关计算 --------------------
    def _get_stock_assets(self, product: str) -> Tuple[float, float]:
        assets = self._read_product_files(product, 'assets', '_assets')
        if assets is None:
            return 0.0, 0.0
        return assets['net_assets'].sum(), assets['ava_cash'].sum()

    def _get_stock_hold_value(self, product: str) -> float:
        pos = self._read_product_files(product, 'pos', '_pos')
        if pos is None or pos.empty:
            return 0.0
        pos = pos[pos['code'].isin(self.stk_price.index)]
        if pos.empty:
            return 0.0
        pos = pos.merge(self.stk_price[['close']], left_on='code', right_index=True, how='left')
        return (pos['close'] * pos['hold']).sum()

    def _deal_stock_order(self, acct: str, adj_fee_ratio=0.00015, t0_fee_ratio=0.0002):
        """处理股票交易记录（调仓/T0）"""
        stk_order = self._read_product_files(acct, 'entrust', '_entrust')
        if stk_order is None or stk_order.empty:
            return 0, 0, 0, 0, 0, 0, 0

        stk_order = stk_order[stk_order['code'].isin(self.stk_price.index)]
        if stk_order.empty:
            return 0, 0, 0, 0, 0, 0, 0

        stk_order = pd.merge(stk_order, self.stk_price.reset_index(), on=['code', 'date'], how='left')
        stk_order['filled_amount'] = stk_order['filled_vol'] * stk_order['filled_price']
        stk_order['fee'] = np.where(stk_order['algo'] == 't0',
                                    np.where(stk_order['dir'] == 1, stk_order['filled_amount'] * t0_fee_ratio,
                                             stk_order['filled_amount'] * (t0_fee_ratio + 0.0005)),
                                    np.where(stk_order['dir'] == 1, stk_order['filled_amount'] * adj_fee_ratio,
                                             stk_order['filled_amount'] * (adj_fee_ratio + 0.0005)))

        # 调仓数据
        adj_data = stk_order[stk_order['algo'] == 'adj'].copy()
        if not adj_data.empty:
            adj_data['adj_benchmark'] = adj_data.apply(lambda row: row.get(f"{row['time_period']}_vwap"), axis=1)
            adj_data['benchmark_slipvwap'] = np.where(
                adj_data['filled_price'] != 0,
                (adj_data['filled_price'] / adj_data['adj_benchmark'] - 1) * 10000 * adj_data['dir'] * -1,
                0
            )
            adj_data['benchmark_slipvwap_amount'] = adj_data['benchmark_slipvwap'] * adj_data['filled_amount']
            stk_lng_adj_amount = adj_data['filled_amount'].sum()
            stk_lng_adj_slipvwap = (adj_data[
                                        'benchmark_slipvwap_amount'].sum() / stk_lng_adj_amount) if stk_lng_adj_amount != 0 else 0
            adj_data['pnl'] = (adj_data['preclose'] - adj_data['filled_price']) * adj_data['filled_vol'] * adj_data[
                'dir']
            adj_pnl = adj_data['pnl'].sum()
            adj_fee = adj_data['fee'].sum()
            stk_lng_adj_trade_act_pnl = adj_pnl - adj_fee
        else:
            stk_lng_adj_amount = 0
            stk_lng_adj_slipvwap = 0
            stk_lng_adj_trade_act_pnl = 0

        # T0数据
        t0_data = stk_order[stk_order['algo'] == 't0'].copy()
        if not t0_data.empty:
            t0_data['cs_f'] = t0_data['filled_amount'] * t0_data['dir'] * -1
            tmp_t0 = t0_data.copy()
            tmp_t0['vol'] = tmp_t0['dir'] * tmp_t0['filled_vol']
            grouped = tmp_t0.groupby('code')['vol'].sum().reset_index()
            t0_close_codes = grouped[grouped['vol'] == 0]['code'].tolist()
            t0_data = t0_data[t0_data['code'].isin(t0_close_codes)]
            t0_pnl = t0_data['cs_f'].sum()
            t0_fee = t0_data['fee'].sum()
            stk_lng_t0_trade_act_amount = t0_data['filled_amount'].sum()
            stk_lng_t0_trade_act_pnl = t0_pnl - t0_fee
            stk_lng_t0_trade_expose_act_pnl = 0
        else:
            stk_lng_t0_trade_act_amount = 0
            stk_lng_t0_trade_act_pnl = 0
            stk_lng_t0_trade_expose_act_pnl = 0

        stk_sht_trade_pnl = 0
        return (stk_lng_adj_amount, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl,
                stk_lng_t0_trade_act_amount, stk_lng_t0_trade_act_pnl,
                stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl)

    def _deal_stock_pos(self, acct: str):
        """处理股票持仓盈亏"""
        stk_pos = self._read_product_files(acct, 'pos', '_pos')
        if stk_pos is None or stk_pos.empty:
            return 0, 0, 0, 0
        stk_pos = stk_pos[stk_pos['code'].isin(self.stk_price.index)]
        if stk_pos.empty:
            return 0, 0, 0, 0
        stk_pos = pd.merge(stk_pos, self.stk_price, left_on='code', right_index=True, how='left')
        stk_pos['hold_pnl'] = stk_pos['hold'] * (stk_pos['close'] - stk_pos['preclose'])
        stk_pos['hold_value'] = stk_pos['close'] * stk_pos['hold']
        stk_lng_hold_value = stk_pos['hold_value'].sum()
        stk_lng_hold_act_pnl = stk_pos['hold_pnl'].sum()
        return stk_lng_hold_value, stk_lng_hold_act_pnl, 0, 0

    # -------------------- 产品净值整合 --------------------
    def calculate_product(self, product: str, info: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """计算单个产品的净值信息、预估净值、期货明细"""
        chinese_name = info.get('product_chinese_name', '')
        product_type = info.get('product_type', '')
        benchmark = info.get('benchmark', 'hs300')

        # 获取各项数据
        stk_hold = self._get_stock_hold_value(product)
        stk_assets, stk_cash = self._get_stock_assets(product)
        fut_assets, fut_withdraw, fut_deposit = self._get_futures_assets(product)
        quant_diff, cta_diff, quant_fut_hold_value, fut_hold_value, margin = self._calc_futures_diff_pnl(product)
        opt_assets, opt_pnl = self._get_options_assets(product)
        deposits = self._get_deposits_withdrawals(product)
        pre_net_close, pre_net_settle, pre_fut_assets = self._get_previous_net(product)

        # 交易和持仓明细（用于期货明细表）
        quant_order_pnl, quant_used_fee, cta_order_pnl, cta_used_fee = self._calc_futures_order_pnl(product)
        quant_hold_val, quant_hold_pnl, quant_diff_pnl, cta_hold_val, cta_hold_pnl, cta_diff_pnl = self._calc_futures_pos_pnl(
            product)

        # 净资产计算
        net_assets_close = stk_assets + fut_assets + opt_assets
        net_assets_settle = stk_assets + fut_assets + quant_diff + cta_diff + opt_assets

        # 增长率
        net_growth_close = (net_assets_close - deposits) / pre_net_close - 1 if pre_net_close != 0 else np.nan
        net_growth_settle = (net_assets_settle - deposits) / pre_net_settle - 1 if pre_net_settle != 0 else np.nan
        benchmark_chg = self.index_df.loc[benchmark, 'chg'] if benchmark in self.index_df.index else np.nan
        if product == 'PBTZ2H':
            fut_diff = cta_diff
        else:
            fut_diff = cta_diff + quant_diff

        # 从邮件获取前一日单位净值
        pre_unit_net = self._get_pre_net_from_mail(chinese_name)
        # pre_unit_net = 0

        # 产品净值表
        product_df = pd.DataFrame([{
            '日期': self.date,
            '产品': product,
            '产品名': chinese_name,
            '出入金': deposits,
            '产品净资产(收盘价)': net_assets_close,
            '产品净资产(结算价)': net_assets_settle,
            '产品昨收净资产(收盘价)': pre_net_close,
            '产品昨收净资产(结算价)': pre_net_settle,
            '证券户净资产': stk_assets,
            '证券户持仓市值': stk_hold,
            '证券户可用现金': stk_cash,
            '期货户静态权益': fut_assets + fut_diff,
            '期货户出金': fut_withdraw,
            '期货户入金': fut_deposit,
            '期货户动态权益': fut_assets,
            '期货户名义市值': fut_hold_value,
            '期货户保证金': margin,
            '期货户风险度': margin / (fut_assets + fut_diff) if (fut_assets + fut_diff) != 0 else 0,
            '仓位': stk_hold / (stk_assets + fut_assets) if (stk_assets + fut_assets) != 0 else 0,
            '对冲比例': abs(stk_hold / quant_fut_hold_value) if quant_fut_hold_value != 0 else (
                np.nan if product_type == 'zx' else 0),
            '期权总资产': opt_assets,
            '基准': benchmark_chg,
            '净值增长': net_growth_close,
            '超额': net_growth_close - benchmark_chg if product_type != 'zx' else np.nan,
            '净值增长(结算价)': net_growth_settle,
            '超额(结算价)': net_growth_settle - benchmark_chg if product_type != 'zx' else np.nan,
        }])

        # 预估净值表
        estimated_df = pd.DataFrame([{
            '日期': self.date,
            '产品名': chinese_name,
            '昨日单位净值': pre_unit_net,
            '净值增长': net_growth_settle,
            '当日预估净值': pre_unit_net * (1 + net_growth_settle) if pre_unit_net != 0 else np.nan,
        }])

        # 期货明细表
        futures_detail_df = pd.DataFrame([{
            '日期': self.date,
            '产品名': chinese_name,
            '昨日期货静态权益': pre_fut_assets,
            '期货户静态权益': fut_assets + fut_diff,
            '期货户出入金': fut_deposit - fut_withdraw,
            '期货户收益': (fut_assets + fut_diff) - pre_fut_assets - (fut_deposit - fut_withdraw),
            '期货户保证金': margin,
            '期货户风险度': margin / (fut_assets + fut_diff) if (fut_assets + fut_diff) != 0 else 0,
            'quant收益': quant_order_pnl - quant_used_fee + quant_hold_pnl,
            'cta收益': cta_order_pnl - cta_used_fee + cta_hold_pnl,
            'quant+cta收益': (quant_order_pnl - quant_used_fee + quant_hold_pnl +
                              cta_order_pnl - cta_used_fee + cta_hold_pnl),
        }])

        return product_df, estimated_df, futures_detail_df

    def calculate_account(self, product: str, chinese_name: str, benchmark: str, acct: str, acct_info: Dict) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """计算单个账户的股票交易和持仓明细"""
        (stk_lng_adj_amount, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl,
         stk_lng_t0_trade_act_amount, stk_lng_t0_trade_act_pnl,
         stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl) = self._deal_stock_order(acct)

        stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl = self._deal_stock_pos(acct)

        benchmark_chg = self.index_df.loc[benchmark, 'chg'] if benchmark in self.index_df.index else 0.0
        stk_assets, stk_cash = self._get_stock_assets(acct)

        cangwei = stk_lng_hold_value / stk_assets if stk_assets != 0 else 0.0
        stk_lng_alpha_tot_pnl = stk_lng_hold_act_pnl + stk_lng_adj_trade_act_pnl
        stk_lng_alpha_tot_ret = stk_lng_alpha_tot_pnl / stk_lng_hold_value if stk_lng_hold_value != 0 else 0.0
        stk_lng_tot_pnl = stk_lng_alpha_tot_pnl + stk_lng_t0_trade_act_pnl
        stk_lng_tot_ret = stk_lng_tot_pnl / stk_lng_hold_value if stk_lng_hold_value != 0 else 0.0

        account_df = pd.DataFrame([{
            '日期': self.date,
            '产品名': chinese_name,
            '产品': product,
            '账户名': acct_info.get('acct_chinese_name'),
            '账户': acct,
            '证券户净资产': stk_assets,
            '证券户可用现金': stk_cash,
            '多头持仓市值': stk_lng_hold_value,
            '多头持仓收益': stk_lng_hold_act_pnl,
            '空头持仓市值': stk_sht_hold_value,
            '空头持仓收益': stk_sht_hold_act_pnl,
            '多头调仓金额': stk_lng_adj_amount,
            '多头调仓收益': stk_lng_adj_trade_act_pnl,
            '多头调仓滑点': stk_lng_adj_slipvwap,
            '多头T0金额': stk_lng_t0_trade_act_amount,
            '多头T0收益': stk_lng_t0_trade_act_pnl,
            '多头T0敞口收益': stk_lng_t0_trade_expose_act_pnl,
            '空头调仓收益': stk_sht_trade_pnl,
            '多头总收益': stk_lng_alpha_tot_pnl,
            '空头总收益': stk_sht_trade_pnl + stk_sht_hold_act_pnl,
            '证券户总收益': stk_lng_alpha_tot_pnl + stk_sht_trade_pnl + stk_sht_hold_act_pnl,
            '多头收益率': stk_lng_alpha_tot_ret,
            '多头收益率(含T0)': stk_lng_tot_ret,
            '基准': benchmark,
            '基准收益率': benchmark_chg,
            # '多头超额': (stk_lng_alpha_tot_ret - benchmark_chg) * cangwei,
            # '多头超额(含T0)': (stk_lng_tot_ret - benchmark_chg) * cangwei,
            '多头超额': (stk_lng_alpha_tot_ret - benchmark_chg),
            '多头超额(含T0)': (stk_lng_tot_ret - benchmark_chg),
        }])

        lng_ava_t0_amount = stk_lng_hold_value - stk_lng_adj_amount * 0.5
        t0_df = pd.DataFrame([{
            '日期': self.date,
            '产品名': chinese_name,
            '产品': product,
            '账户名': acct_info.get('acct_chinese_name'),
            '账户': acct,
            't0_algo': acct_info.get('t0_algo'),
            '证券户可用现金': stk_cash,
            '多头持仓市值': stk_lng_hold_value,
            '多头可T0总金额': lng_ava_t0_amount,
            '多头T0金额': stk_lng_t0_trade_act_amount,
            '多头T0换手率(双边)': stk_lng_t0_trade_act_amount / lng_ava_t0_amount if lng_ava_t0_amount != 0 else 0.0,
            '多头T0收益': stk_lng_t0_trade_act_pnl,
            '多头T0敞口收益': stk_lng_t0_trade_expose_act_pnl,
            '多头T0收益率': stk_lng_t0_trade_act_pnl / lng_ava_t0_amount if lng_ava_t0_amount != 0 else 0.0,
            'adj_algo': acct_info.get('adj_algo'),
            '多头调仓金额': stk_lng_adj_amount,
            '多头调仓收益': stk_lng_adj_trade_act_pnl,
            '多头调仓滑点': stk_lng_adj_slipvwap,
        }])

        return account_df, t0_df


# ==================== 报表展示与科学可视化 ====================
class ReportGenerator:
    """报表生成器（科学展示）"""

    @staticmethod
    def format_percent(x: float) -> str:
        return f"{x:.2%}" if pd.notnull(x) else "--"

    @staticmethod
    def format_money(x: float, unit: str = '万') -> str:
        if unit == '万':
            return f"{x / 10000:,.2f}"
        return f"{x:,.2f}"

    @staticmethod
    def display_product_summary(df: pd.DataFrame):
        """展示产品净值汇总表"""
        display_df = df[[
            '日期', '产品名', '出入金', '产品净资产(结算价)', '证券户净资产',
            '证券户持仓市值', '仓位', '对冲比例', '基准', '净值增长(结算价)', '超额(结算价)'
        ]].copy()
        display_df['出入金'] = display_df['出入金'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['产品净资产(结算价)'] = display_df['产品净资产(结算价)'].apply(
            lambda x: ReportGenerator.format_money(x))
        display_df['证券户净资产'] = display_df['证券户净资产'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['证券户持仓市值'] = display_df['证券户持仓市值'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['仓位'] = display_df['仓位'].apply(ReportGenerator.format_percent)
        display_df['对冲比例'] = display_df['对冲比例'].apply(ReportGenerator.format_percent)
        display_df['基准'] = display_df['基准'].apply(ReportGenerator.format_percent)
        display_df['净值增长(结算价)'] = display_df['净值增长(结算价)'].apply(ReportGenerator.format_percent)
        display_df['超额(结算价)'] = display_df['超额(结算价)'].apply(ReportGenerator.format_percent)

        print("\n" + "=" * 80)
        print("产品净值汇总（单位：万元）")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    @staticmethod
    def display_futures_summary(df: pd.DataFrame):
        """展示期货账户汇总"""
        if df.empty:
            return
        display_df = df.copy()
        for col in ['昨日期货静态权益', '期货户静态权益', '期货户出入金', '期货户收益', '期货户保证金', 'quant收益',
                    'cta收益', 'quant+cta收益']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: ReportGenerator.format_money(x))
        if '期货户风险度' in display_df.columns:
            display_df['期货户风险度'] = display_df['期货户风险度'].apply(ReportGenerator.format_percent)
        print("\n" + "=" * 80)
        print("期货账户明细（单位：万元）")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    @staticmethod
    def display_estimated_nav(df: pd.DataFrame):
        """展示预估净值"""
        if df.empty:
            return
        display_df = df.copy()
        display_df['净值增长'] = display_df['净值增长'].apply(ReportGenerator.format_percent)
        display_df['当日预估净值'] = display_df['当日预估净值'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")
        print("\n" + "=" * 80)
        print("预估净值")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    @staticmethod
    def display_account_summary(df: pd.DataFrame):
        """展示账户收益明细"""
        if df.empty:
            return
            # 多头T0收益
        display_df = df[
            ['日期', '产品名', '账户名', '证券户净资产', '多头持仓市值', '证券户总收益', '基准收益率', '多头超额',
             '多头超额(含T0)']].copy()
        tmp = df[
            ['日期', '产品名', '账户名', '证券户净资产', '多头持仓市值', '证券户总收益', '多头T0收益', '基准收益率',
             '多头超额', '多头超额(含T0)']].copy()
        tmp = tmp[tmp['产品名'] == '配邦恒升中性1号'].copy()
        if isinstance(tmp, pd.DataFrame) and (not tmp.empty):
            tmp = tmp.groupby(['日期', '产品名']).agg(
                {'证券户净资产': 'sum', '多头持仓市值': 'sum', '证券户总收益': 'sum', '多头T0收益': 'sum',
                 '基准收益率': 'mean', '多头超额': 'mean', '多头超额(含T0)': 'mean'}).reset_index()
            tmp['账户名'] = '配邦恒升合计'
            tmp['多头超额'] = tmp['证券户总收益'] / tmp['多头持仓市值'] - tmp['基准收益率']
            tmp['多头超额(含T0)'] = (tmp['证券户总收益'] + tmp['多头T0收益']) / tmp['多头持仓市值'] - tmp['基准收益率']
            display_df = pd.concat([display_df, tmp[
                ['日期', '产品名', '账户名', '证券户净资产', '多头持仓市值', '证券户总收益', '基准收益率', '多头超额',
                 '多头超额(含T0)']]])
        display_df['证券户净资产'] = display_df['证券户净资产'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['多头持仓市值'] = display_df['多头持仓市值'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['证券户总收益'] = display_df['证券户总收益'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['基准收益率'] = display_df['基准收益率'].apply(ReportGenerator.format_percent)
        display_df['多头超额'] = display_df['多头超额'].apply(ReportGenerator.format_percent)
        display_df['多头超额(含T0)'] = display_df['多头超额(含T0)'].apply(ReportGenerator.format_percent)
        print("\n" + "=" * 80)
        print("证券账户超额收益明细")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

    @staticmethod
    def display_t0_summary(df: pd.DataFrame):
        """展示T0及调仓滑点明细"""
        if df.empty:
            return
        display_df = df[
            ['日期', '产品名', '账户名', 't0_algo', '多头T0换手率(双边)', '多头T0收益', '多头T0收益率', 'adj_algo',
             '多头调仓滑点']].copy()
        display_df['多头T0收益'] = display_df['多头T0收益'].apply(lambda x: ReportGenerator.format_money(x))
        display_df['多头T0换手率(双边)'] = display_df['多头T0换手率(双边)'].apply(ReportGenerator.format_percent)
        display_df['多头T0收益率'] = display_df['多头T0收益率'].apply(ReportGenerator.format_percent)
        display_df['多头调仓滑点'] = display_df['多头调仓滑点'].apply(lambda x: f"{x:.2f}")
        print("\n" + "=" * 80)
        print("T0交易及调仓滑点明细")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))


# ==================== 主流程 ====================
def main():
    # 读取配置
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, 'split_system.yaml')
    config_mgr = ConfigManager(config_path)
    config = config_mgr.load()
    date = config_mgr.get_date()

    # 可手动指定日期（原代码中有硬编码，保留灵活性）
    # date = '20260520'   # 如需固定日期可取消注释

    # 判断是否为交易日
    ddp = DateDealProcess()
    if not ddp.judge_trading_date(date):
        return

    # 初始化rqdatac
    rqdatac_config = config.get('rqdatac')
    rqdatac.init(
        rqdatac_config.get('rqdatac_user', '13601611030'),
        rqdatac_config.get('rqdatac_password', 'PB123456789')
    )

    calculator = ProductNetCalculator(date, config)

    all_product_nav = []
    all_estimated_nav = []
    all_futures_detail = []
    all_account_detail = []
    all_t0_detail = []

    for product, info in calculator.product_info.items():
        logger.info(f"计算产品: {product}")
        try:
            nav_df, est_df, fut_df = calculator.calculate_product(product, info)
            all_product_nav.append(nav_df)
            # if product != 'PBZS1H':   # 排除特定产品
            #     all_estimated_nav.append(est_df)
            #     all_futures_detail.append(fut_df)

            if product != 'PBZS1H':  # 排除特定产品
                all_estimated_nav.append(est_df)
            all_futures_detail.append(fut_df)

            # 保存单个产品净值文件
            out_dir = os.path.join(calculator.out_path, 'net_data', date)
            os.makedirs(out_dir, exist_ok=True)
            nav_df.to_csv(os.path.join(out_dir, f"{date}_{product}_net_info.csv"), index=False)
            logger.info(f"保存 {product} 净值文件")

            # 保存单个产品期货文件
            fut_out_dir = os.path.join(calculator.out_path, 'stk_fut', date)
            os.makedirs(fut_out_dir, exist_ok=True)
            fut_df.to_csv(os.path.join(fut_out_dir, f"{date}_{product}_fut_info.csv"), index=False)
            logger.info(f"保存 {product} 期货文件")

            # 计算账户明细
            for acct, acct_info in info.get('acct_info', {}).items():
                try:
                    acc_df, t0_df = calculator.calculate_account(
                        product, info.get('product_chinese_name', ''),
                        info.get('benchmark', 'hs300'), acct, acct_info
                    )
                    all_account_detail.append(acc_df)
                    all_t0_detail.append(t0_df)
                    # 保存账户明细
                    stk_out_dir = os.path.join(calculator.out_path, 'stk_fut', date)
                    os.makedirs(stk_out_dir, exist_ok=True)
                    acc_df.to_csv(os.path.join(stk_out_dir, f"{date}_{acct}_stk_info.csv"), index=False)

                    # 保存账户t0调仓明细
                    t0_out_dir = os.path.join(calculator.out_path, 't0_adj', date)
                    os.makedirs(t0_out_dir, exist_ok=True)
                    t0_df.to_csv(os.path.join(t0_out_dir, f"{date}_{acct}_t0_adj_info.csv"), index=False)
                except Exception as e:
                    logger.error(f"计算账户 {acct} 失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"计算产品 {product} 失败: {e}", exc_info=True)

    # 合并并展示结果
    if all_product_nav:
        combined_nav = pd.concat(all_product_nav, ignore_index=True)
        ReportGenerator.display_product_summary(combined_nav)
        # 保存汇总文件
        combined_nav.to_csv(os.path.join(calculator.out_path, 'net_data', date, f"{date}_all_product_info.csv"),
                            index=False)
        # 同步输出文件到本地
        try:
            for sync_dir in ['net_data', 'stk_fut', 't0_adj']:
                src_sync = os.path.join(calculator.out_path, sync_dir, date)
                dst_sync = os.path.join(calculator.local_out_path, sync_dir, date)
                if os.path.exists(src_sync):
                    shutil.copytree(src_sync, dst_sync, dirs_exist_ok=True)
                    logger.debug(f'  同步 {sync_dir} 到本地')
        except Exception as sync_e:
            logger.warning(f'  同步到本地失败: {sync_e}')

    if all_futures_detail:
        combined_fut = pd.concat(all_futures_detail, ignore_index=True)
        ReportGenerator.display_futures_summary(combined_fut)

    if all_account_detail:
        combined_acc = pd.concat(all_account_detail, ignore_index=True)
        ReportGenerator.display_account_summary(combined_acc)

    if all_t0_detail:
        combined_t0 = pd.concat(all_t0_detail, ignore_index=True)
        ReportGenerator.display_t0_summary(combined_t0)

    if all_estimated_nav:
        combined_est = pd.concat(all_estimated_nav, ignore_index=True)
        ReportGenerator.display_estimated_nav(combined_est)


def run():
    # 读取配置
    root_path = os.path.dirname(os.path.abspath(__file__))
    start_date = '20250922'  # 20250725
    end_date = '20260413'
    dates = rqdatac.get_trading_dates(start_date=start_date, end_date=end_date)
    dates = [str(_date).replace('-', '') for _date in dates]

    for date in dates:
        logger.info(f"计算日期为：{date}")
        config_path = f'/home/zhanggh/DailyScripts/split_config/{date}_split_system.yaml'
        # config_path = os.path.join(root_path, 'split_system.yaml')
        config_mgr = ConfigManager(config_path)
        config = config_mgr.load()
        # date = config_mgr.get_date()

        # 可手动指定日期
        # date = '20260410'

        # 初始化rqdatac
        rqdatac.init(
            config.get('rqdatac_user', '13601611030'),
            config.get('rqdatac_password', 'PB123456789')
        )

        calculator = ProductNetCalculator(date, config)

        all_product_nav = []
        all_estimated_nav = []
        all_futures_detail = []
        all_account_detail = []
        all_t0_detail = []

        for product, info in calculator.product_info.items():
            logger.info(f"计算产品: {product}")
            try:
                nav_df, est_df, fut_df = calculator.calculate_product(product, info)
                all_product_nav.append(nav_df)
                if product != 'PBZS1H':  # 排除特定产品
                    all_estimated_nav.append(est_df)
                    all_futures_detail.append(fut_df)

                # 保存单个产品净值文件
                out_dir = os.path.join(calculator.out_path, 'net_data', date)
                os.makedirs(out_dir, exist_ok=True)
                nav_df.to_csv(os.path.join(out_dir, f"{date}_{product}_net_info.csv"), index=False)
                logger.info(f"保存 {product} 净值文件")

                # 保存单个产品期货文件
                fut_out_dir = os.path.join(calculator.out_path, 'stk_fut', date)
                os.makedirs(fut_out_dir, exist_ok=True)
                fut_df.to_csv(os.path.join(fut_out_dir, f"{date}_{product}_fut_info.csv"), index=False)
                logger.info(f"保存 {product} 期货文件")

                # 计算账户明细
                for acct, acct_info in info.get('acct_info', {}).items():
                    try:
                        acc_df, t0_df = calculator.calculate_account(
                            product, info.get('product_chinese_name', ''),
                            info.get('benchmark', 'hs300'), acct, acct_info
                        )
                        all_account_detail.append(acc_df)
                        all_t0_detail.append(t0_df)
                        # 保存账户明细
                        stk_out_dir = os.path.join(calculator.out_path, 'stk_fut', date)
                        os.makedirs(stk_out_dir, exist_ok=True)
                        acc_df.to_csv(os.path.join(stk_out_dir, f"{date}_{acct}_stk_info.csv"), index=False)

                        # 保存账户t0调仓明细
                        t0_out_dir = os.path.join(calculator.out_path, 't0_adj', date)
                        os.makedirs(t0_out_dir, exist_ok=True)
                        t0_df.to_csv(os.path.join(t0_out_dir, f"{date}_{acct}_t0_adj_info.csv"), index=False)
                    except Exception as e:
                        logger.error(f"计算账户 {acct} 失败: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"计算产品 {product} 失败: {e}", exc_info=True)

        # 合并并展示结果
        if all_product_nav:
            combined_nav = pd.concat(all_product_nav, ignore_index=True)
            ReportGenerator.display_product_summary(combined_nav)
            # 保存汇总文件
            combined_nav.to_csv(
                os.path.join(calculator.standard_path, 'net_data', date, f"{date}_all_product_info.csv"), index=False)
            # 同步输出文件到本地
            try:
                for sync_dir in ['net_data', 'stk_fut', 't0_adj']:
                    src_sync = os.path.join(calculator.out_path, sync_dir, date)
                    dst_sync = os.path.join(calculator.local_out_path, sync_dir, date)
                    if os.path.exists(src_sync):
                        shutil.copytree(src_sync, dst_sync, dirs_exist_ok=True)
                        logger.debug(f'  同步 {sync_dir} 到本地')
            except Exception as sync_e:
                logger.warning(f'  同步到本地失败: {sync_e}')

        if all_futures_detail:
            combined_fut = pd.concat(all_futures_detail, ignore_index=True)
            ReportGenerator.display_futures_summary(combined_fut)

        if all_account_detail:
            combined_acc = pd.concat(all_account_detail, ignore_index=True)
            ReportGenerator.display_account_summary(combined_acc)

        if all_t0_detail:
            combined_t0 = pd.concat(all_t0_detail, ignore_index=True)
            ReportGenerator.display_t0_summary(combined_t0)

        if all_estimated_nav:
            combined_est = pd.concat(all_estimated_nav, ignore_index=True)
            ReportGenerator.display_estimated_nav(combined_est)


if __name__ == '__main__':
    main()
    # run()