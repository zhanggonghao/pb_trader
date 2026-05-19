"""
产品净值计算脚本
根据原始持仓、资产等数据，计算产品净值、收益、风险指标等。
"""

import logging
import os
import sys
import warnings
from typing import Dict, Optional, Tuple, List
from tabulate import tabulate
from glob import glob
from ultis.email_manager import *
import pandas as pd
import numpy as np
import datetime as dt
import yaml
import rqdatac

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 忽略 pandas 警告
warnings.filterwarnings('ignore')


class ProductNetCalculator:
    """产品净值计算器"""

    # 保证金比例（可配置）
    DEFAULT_MARGIN_RATIO = 0.12

    def __init__(self, date: str, config: Dict):
        """
        初始化计算器

        Args:
            date: 计算日期，格式 YYYYMMDD
            config: 配置字典
        """
        self.date = date
        self.pre_date = self._get_previous_trading_date(date)
        self.standard_path = config.get('standard_path')
        self.product_info = config.get('product_info', {})
        self.margin_ratio = config.get('margin_ratio', self.DEFAULT_MARGIN_RATIO)
        self.net_email_path = config.get('net_email_path') # 净值邮件目录
        self._download_mail()

        # # 获取市场数据
        self.index_df = self._get_benchmark_info()
        self.fut_info = self._get_fut_info()
        # self.stk_close = self._get_stk_close()
        self.stk_close = self._get_stk_price_info()
        # self.stk_close.to_csv('/home/zhanggh/DailyScripts/20260409_stk_close.csv')
    
    # --------------------------- 配置文件 ---------------------------
    def _get_adj_periods(self):
        adj_periods = []
        for product in self.product_info:
            for acct in self.product_info.get(product).get('acct_info'):
                acct_info = self.product_info.get(product).get('acct_info').get(acct)
                adj_algo_starttime = acct_info.get('adj_algo_starttime')
                adj_algo_endtime = acct_info.get('adj_algo_endtime')
                adj_periods += [(adj_algo_starttime, adj_algo_endtime)]

        return list(set(adj_periods))

    # --------------------------- 工具方法 ---------------------------
    @staticmethod
    def _get_previous_trading_date(date: str) -> str:
        """获取前一个交易日"""
        try:
            return rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
        except Exception as e:
            logger.error(f"获取前一个交易日失败: {e}")
            return date  # 保底，实际使用时需确保日期有效

    @staticmethod
    def match_file_v2(path, key, date):
        """匹配包含日期和关键字的文件"""
        pattern = os.path.join(path, f"*{date}*")
        for file in glob(pattern):
            if key in file:
                return file
        return None
    
    # ---------------------------- 下载邮件 ---------------------------
    def _download_mail(self):
        product_chinese_names = [info.get('product_chinese_name') for key, info in self.product_info.items()] 
        # ['配邦投资二号', '配邦中圣1号', '配邦恒升中性1号']
        format_predate = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'
        print(format_predate)
        manager = EmailManager()
        for product_chinese_name in product_chinese_names:
            if product_chinese_name == '配邦中圣1号':
                product_chinese_name = '配邦中圣'
            match_file = self.match_file_v2(self.net_email_path, product_chinese_name, format_predate)
            if (not match_file) or (not os.path.exists(match_file)):
                try:
                    manager.download_attachments_by_keyword([product_chinese_name, format_predate], save_dir=self.net_email_path, file_extensions=['.xlsx'])
                except Exception as e:
                    logger.warning(f"下载 {self.pre_date} {product_chinese_name}产品净值邮件出错: {e}")

        manager.logout()

    # --------------------------- 市场数据获取 ---------------------------
    def _get_benchmark_info(self) -> pd.DataFrame:
        """
        获取基准指数数据（沪深300、中证500、中证A500、中证1000）
        返回 DataFrame，索引为 benchmark 名称，包含收盘价、前收盘、涨跌幅
        """
        codes = ['000300.XSHG', '000905.XSHG', '000510.XSHG', '000852.XSHG']
        benchmarks = ['hs300', 'zz500', 'a500', 'zz1000']

        try:
            # 尝试获取当日数据
            df = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date,
                frequency='1d', fields=None, adjust_type='pre',
                skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
        except Exception as e:
            logger.warning(f"获取指数日线数据失败 ({self.date}): {e}，尝试使用前一日收盘+实时行情")
            # 使用前一日收盘价 + 实时行情
            prev_df = rqdatac.get_price(
                codes, start_date=self.pre_date, end_date=self.pre_date,
                frequency='1d', fields=None, adjust_type='pre',
                skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
            prev_df = prev_df[['order_book_id', 'close']].rename(columns={'close': 'prev_close'})

            cur_df = rqdatac.current_minute(codes, skip_suspended=False).reset_index()
            cur_df = cur_df[['order_book_id', 'close']]

            df = pd.merge(prev_df, cur_df, on='order_book_id')
            df = df[['order_book_id', 'close', 'prev_close']]

        df = df.rename(columns={'order_book_id': 'code'})
        df['chg'] = df['close'] / df['prev_close'] - 1
        df['benchmark'] = df['code'].apply(lambda x: benchmarks[codes.index(x)])
        df.set_index('benchmark', inplace=True)
        return df[['close', 'prev_close', 'chg']]

    def _get_fut_info(self) -> pd.DataFrame:
        """
        获取期货合约信息（收盘价、结算价、乘数等）
        返回 DataFrame，索引为合约代码
        """
        # 筛选交易所和品种
        fut_mkt_df = rqdatac.all_instruments(type='Future', market='cn', date=self.date)
        # 只保留中金所和DCE的特定品种, , 'T', 'A', 'm', 'M'
        fut_mkt_df = fut_mkt_df[
            (fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))) &
            ((fut_mkt_df['exchange'] == 'CFFEX') | (fut_mkt_df['exchange'] == 'DCE')) &
            ((fut_mkt_df['symbol'].apply(lambda x: '连续' not in x)) |
             (fut_mkt_df['symbol'].apply(lambda x: '国债' in x)))
        ].reset_index(drop=True)

        codes = fut_mkt_df['order_book_id'].tolist()
        fut_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(
            columns={'order_book_id': 'code'}
        )

        try:
            # 尝试获取当日日线数据
            price_df = rqdatac.get_price(codes, start_date=self.date, end_date=self.date,frequency='1d', fields=None, adjust_type='pre',skip_suspended=False, market='cn', expect_df=True).reset_index()
            price_df = price_df[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
            fut_df = pd.merge(fut_df, price_df, on='code', how='left')
        except Exception as e:
            logger.warning(f"获取期货日线数据失败 ({self.date}): {e}，尝试使用前一日日线+分钟线计算")
            # 使用前一日日线数据
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
            minute_file = f'/home/zhanggh/testscripts/test_data/{self.date}_data.csv'
            if not os.path.exists(minute_file):
                raise FileNotFoundError(f"分钟线数据文件不存在: {minute_file}")

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
            fut_df['settlement'] = fut_df['settlement'] / fut_df['contract_multiplier']  # 转换为单位价格

        fut_df.set_index('code', inplace=True)
        return fut_df

    def _get_stk_close(self) -> pd.DataFrame:
        """获取股票当日收盘价，返回 DataFrame，索引为股票代码（int）"""
        all_stk = rqdatac.all_instruments(type='CS', market='cn', date=self.date)
        codes = all_stk['order_book_id'].unique().tolist()

        try:
            df = rqdatac.get_price(
                codes, start_date=self.date, end_date=self.date,
                frequency='1d', fields=None, adjust_type='pre',
                skip_suspended=False, market='cn', expect_df=True
            ).reset_index()
        except Exception as e:
            logger.warning(f"获取股票日线数据失败 ({self.date}): {e}，使用实时行情")
            df = rqdatac.current_minute(codes, skip_suspended=False).reset_index()

        df = df[['order_book_id', 'close']].rename(columns={'order_book_id': 'code'})
        df['code'] = df['code'].apply(lambda x: int(x.split('.')[0]))
        df.set_index('code', inplace=True)
        return df

    # 获取股票当天全市场的vwap数据，包含昨收价，今收价，开盘半小时的vwap，全天vwap，以及调仓交易时段的vwap
    def _get_stk_price_info(self):
        add_time_periods = self._get_adj_periods()
        time_periods = list(set([('093000', '093500')] + add_time_periods))
        print(time_periods)

        all_stk = rqdatac.all_instruments(type='CS', market='cn', date=self.date)
        codes = all_stk['order_book_id'].unique().tolist()
        try:
            price_info = rqdatac.get_price(codes, start_date=self.date, end_date=self.date, frequency='1d', fields=[
                                        'prev_close', 'close', 'total_turnover', 'volume'], adjust_type='pre', skip_suspended=False, market='cn', expect_df=True, time_slice=None).reset_index()
            price_info['allday_vwap'] = np.where(
                price_info['volume'] > 0, price_info['total_turnover'] / price_info['volume'], price_info['close'])
            price_info['date'] = price_info['date'].dt.strftime('%Y%m%d')
            price_info = price_info[['order_book_id','date', 'prev_close', 'close', 'allday_vwap']]

            stk_mkt_df = rqdatac.get_price(codes, start_date=self.date, end_date=self.date, frequency='1m', fields=None,
                                        adjust_type='pre', skip_suspended=False, market='cn', expect_df=True, time_slice=None).reset_index()

            stk_mkt_df['date'] = stk_mkt_df['datetime'].dt.strftime('%Y%m%d')
            stk_mkt_df['time'] = stk_mkt_df['datetime'].dt.strftime('%H%M%S')

            for i in time_periods:
                st = i[0]
                et = i[1]
                label = f'{st}-{et}_vwap'
                tmp = stk_mkt_df[stk_mkt_df['time'].between(
                    st, et, inclusive='right')]
                grouped = tmp[['order_book_id', 'date', 'total_turnover', 'volume']].groupby(
                    ['order_book_id', 'date']).sum().reset_index()
                grouped[label] = np.where(
                    grouped['volume'] > 0, grouped['total_turnover'] / grouped['volume'], 0)
                price_info = pd.merge(price_info, grouped[['order_book_id', 'date', label]], on=['order_book_id', 'date'])

        except Exception as e:
            logger.warning(f"获取股票数据失败 ({self.date}): {e}，日线数据使用实时行情，vwap数据填为0.0")
            stk_today = rqdatac.current_minute(codes, fields = ['close'], skip_suspended=False).reset_index()
            stk_preday = rqdatac.get_price(codes, start_date=self.pre_date, end_date=self.pre_date, frequency='1d',
                                            fields=['close'], adjust_type='pre', skip_suspended=False, market='cn', expect_df=True, time_slice=None).reset_index().rename(columns={'close': 'prev_close'})
            price_info = pd.merge(stk_today[['order_book_id', 'close']], stk_preday[['order_book_id', 'prev_close']])
            price_info['date'] = self.date
            price_info['allday_vwap'] = 0.0
            price_info = price_info[['order_book_id','date', 'prev_close', 'close', 'allday_vwap']]
            
            for i in time_periods:
                st = i[0]
                et = i[1]
                label = f'{st}-{et}_vwap'
                price_info[label] = 0.0
            
        price_info = price_info.rename(columns={'order_book_id': 'code', 'prev_close': 'preclose'})
        price_info['code'] = price_info['code'].apply(lambda x: int(str(x)[:6]))
        price_info['date'] = price_info['date'].apply(lambda x: int(str(x).replace('-', '')))
        price_info.set_index('code', inplace=True)
        return price_info

    # --------------------------- 数据读取辅助方法 ---------------------------
    def _read_csv_if_exists(self, path: str, **kwargs) -> Optional[pd.DataFrame]:
        """读取 CSV 文件，如果不存在则返回 None"""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                logger.error(f"读取文件失败 {path}: {e}")
        return None

    def _read_product_files(self, product: str, file_type: str, suffix: str = '') -> Optional[pd.DataFrame]:
        """
        读取产品相关的标准数据文件

        Args:
            product: 产品名
            file_type: 文件类型（fut_data, assets, pos, net_data 等）
            suffix: 文件名后缀（如 _fut_pos.csv）
        Returns:
            DataFrame 或 None
        """
        # /home/zhanggh/DailyScripts/TradeData/standarddata/entrust/20250819/20250819_haitong_PBPFZX1H_entrust.csv
        # pos_files = glob(f'{self.standard_path}/pos/{self.date}/{self.date}_*_{product}_pos.csv')
        # pattern = f"{self.standard_path}/{file_type}/{self.date}/{self.date}_*_{product}{suffix}.csv"
        # print(pattern)
        if file_type in ['assets', 'pos']:
            pattern = f"{self.standard_path}/{file_type}/{self.date}/{self.date}*{product}{suffix}.csv"
            # 这些可能有多文件，用 glob 匹配
            import glob
            files = glob.glob(pattern)
            if not files:
                return None
            dfs = []
            for f in files:
                df = self._read_csv_if_exists(f)
                if df is not None:
                    dfs.append(df)
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return None
        else:
            pattern = f"{self.standard_path}/{file_type}/{self.date}/{self.date}_{product}{suffix}.csv"
            return self._read_csv_if_exists(pattern)

    # --------------------------- 业务逻辑方法 ---------------------------
    def _get_fut_diff_pnl(self, product: str) -> Tuple[float, float, float]:
        """
        计算期货结算价与收盘价的差异盈亏、持仓市值和保证金
        Returns:
            (diff_pnl, hold_value, margin)
        """
        fut_pos = self._read_product_files(product, 'fut_data', '_fut_pos')
        if fut_pos is None:
            return 0.0, 0.0, 0.0

        # 确保合约代码在 fut_info 中
        fut_pos = fut_pos[fut_pos['code'].isin(self.fut_info.index)]
        if fut_pos.empty:
            return 0.0, 0.0, 0.0

        # 合并期货信息
        fut_pos = fut_pos.merge(
            self.fut_info[['close', 'settlement', 'presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )

        # 计算差异盈亏
        fut_pos['diff_pnl'] = (fut_pos['settlement'] - fut_pos['close']) * \
                              fut_pos['Direction'] * fut_pos['vol'] * fut_pos['contract_multiplier']

        # 计算持仓市值（仅 quant 用户）
        fut_pos['amount'] = fut_pos['Direction'] * fut_pos['vol'] * fut_pos['close'] * fut_pos['contract_multiplier']
        fut_pos['settlement_amount'] = fut_pos['contract_multiplier'] * fut_pos['settlement'] * fut_pos['Direction'] * fut_pos['vol']
        fut_pos['lngsht_amount'] = fut_pos['Direction'] * fut_pos['amount']

        # quant名义持仓市值，
        quant_lngsht_value = fut_pos[fut_pos['user'] == 'quant']['lngsht_amount'].sum() if 'user' in fut_pos else 0.0
        quant_hold_value = fut_pos[fut_pos['user'] == 'quant']['amount'].sum() if 'user' in fut_pos else 0.0
        quant_diff_pnl = fut_pos[fut_pos['user'] == 'quant']['diff_pnl'].sum() if 'user' in fut_pos else 0.0
        quant_margin = abs(fut_pos[fut_pos['user'] == 'quant']['settlement_amount'].sum()) * self.margin_ratio  if 'user' in fut_pos else 0.0

        # cta名义持仓市值
        cta_fut_pos = fut_pos[fut_pos['user'] == 'cta'].reset_index(drop=True)
        cta_lngsht_value = 0.0
        cta_hold_value = 0.0
        cta_diff_pnl = 0.0
        cta_margin = 0.0
        if not cta_fut_pos.empty:
            cta_lngsht_value = cta_fut_pos['lngsht_amount'].sum()
            cta_hold_value = cta_fut_pos['amount'].sum()
            cta_diff_pnl = cta_fut_pos['diff_pnl'].sum()
            cta_tmp = cta_fut_pos[['code', 'settlement_amount']].copy()
            cta_tmp['cal_tmp'] = cta_tmp['settlement_amount'].abs()
            margin_tmp = cta_tmp.groupby('code').sum().reset_index()
            margin_tmp['settlement_amount'] = np.where(margin_tmp['settlement_amount'] == 0, margin_tmp['cal_tmp'] * 0.5, margin_tmp['settlement_amount'])
            margin_tmp['settlement_amount'] = margin_tmp['settlement_amount'].abs()
            cta_margin = abs(margin_tmp['settlement_amount'].sum()) * self.margin_ratio

        # 计算保证金
        if product == 'PBTZ2H':
            margin = cta_margin
            diff_pnl = quant_diff_pnl + cta_diff_pnl
        else:
            margin = quant_margin + cta_margin
            diff_pnl = quant_diff_pnl + cta_diff_pnl

        return diff_pnl, quant_hold_value, margin

    # 处理期货交易记录，量化交易收益，cta交易收益
    def _deal_fut_order(self, product):
        fut_order = self._read_product_files(product, 'fut_data', '_fut_order')
        if fut_order is None:
            return 0.0, 0.0, 0.0, 0.0

        # 确保合约代码在 fut_info 中
        fut_order = fut_order[fut_order['code'].isin(self.fut_info.index)]
        if fut_order.empty:
            return 0.0, 0.0, 0.0, 0.0

        # 合并期货信息
        fut_order = fut_order.merge(
            self.fut_info[['close', 'settlement', 'presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )
        quant_order_pnl = 0
        quant_used_fee = 0
        cta_order_pnl = 0
        cta_used_fee = 0
        quant_data = fut_order[fut_order['user'] == 'quant'].copy().reset_index(drop=True)
        if quant_data.shape[0] != 0:
            quant_data['order_pnl'] = quant_data['Direction'] * quant_data['filled_vol'] *  quant_data['contract_multiplier'] * (quant_data['presettlement'] - quant_data['filled_price'])
            # quant_data['order_pnl'] = quant_data['Direction'] * quant_data['filled_vol'] *  quant_data['contract_multiplier'] * (quant_data['preclose'] - quant_data['filled_price'])
            quant_used_fee = quant_data['UsedFee'].sum()
            quant_order_pnl = quant_data['order_pnl'].sum()

        cta_data = fut_order[fut_order['user'] == 'cta'].copy().reset_index(drop=True)
        if cta_data.shape[0] != 0:
            # print(cta_data)
            cta_data['order_pnl'] = cta_data['Direction'] * cta_data['filled_vol'] *  cta_data['contract_multiplier'] * (cta_data['presettlement'] - cta_data['filled_price'])
            # cta_data['order_pnl'] = cta_data['Direction'] * cta_data['filled_vol'] *  cta_data['contract_multiplier'] * (cta_data['preclose'] - cta_data['filled_price'])
            cta_used_fee = cta_data['UsedFee'].sum()
            cta_order_pnl = cta_data['order_pnl'].sum()
            # if date == '20260210':
            #     cta_order_pnl = cta_order_pnl + 3600 if product == 'PBHSZX1H' else cta_order_pnl if product == 'PBPFZX1H' else cta_order_pnl + 7200 
        return round(quant_order_pnl, 2), round(quant_used_fee, 2), round(cta_order_pnl, 2), round(cta_used_fee, 2)

    # 处理期货持仓信息
    def _deal_fut_pos(self, product):
        fut_pos = self._read_product_files(product, 'fut_data', '_fut_pos')
        if fut_pos is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 确保合约代码在 fut_info 中
        fut_pos = fut_pos[fut_pos['code'].isin(self.fut_info.index)]
        if fut_pos.empty:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 合并期货信息
        fut_pos = fut_pos.merge(
            self.fut_info[['close', 'settlement', 'presettlement', 'contract_multiplier']],
            left_on='code', right_index=True, how='left'
        )
        quant_hold_value = 0
        quant_hold_pnl = 0
        quant_diff_pnl = 0
        cta_hold_value = 0
        cta_hold_pnl = 0
        cta_diff_pnl = 0
        quant_data = fut_pos[fut_pos['user'] == 'quant'].reset_index(drop=True)
        if quant_data.shape[0] != 0:
            quant_data['hold_value'] = quant_data['Direction'] * quant_data['vol'] * quant_data['contract_multiplier'] * quant_data['settlement']
            quant_hold_value = quant_data['hold_value'].sum().round(0)
            quant_data['hold_pnl'] = (quant_data['settlement'] - quant_data['presettlement']) * quant_data['vol'] * quant_data['contract_multiplier'] * quant_data['Direction']
            quant_hold_pnl = quant_data['hold_pnl'].sum().round(0)
            quant_data['diff_pnl'] = (quant_data['settlement'] - quant_data['close']) * quant_data['Direction'] * quant_data['vol'] * quant_data['contract_multiplier']
            quant_diff_pnl = quant_data['diff_pnl'].sum().round(0)
        cta_data = fut_pos[fut_pos['user'] == 'cta'].reset_index(drop=True)
        if cta_data.shape[0] != 0:
            cta_data['hold_value'] = cta_data['Direction'] * cta_data['vol'] * cta_data['contract_multiplier'] * cta_data['settlement']
            cta_hold_value = cta_data['hold_value'].sum().round(0)
            cta_data['hold_pnl'] = (cta_data['settlement'] - cta_data['presettlement']) * cta_data['vol'] * cta_data['contract_multiplier'] * cta_data['Direction']
            # cta_data['hold_pnl'] = (cta_data['close'] - cta_data['preclose']) * cta_data['vol'] * cta_data['contract_multiplier'] * cta_data['Direction']
            cta_hold_pnl = cta_data['hold_pnl'].sum().round(0)
            cta_data['diff_pnl'] = (cta_data['settlement'] - cta_data['close']) * cta_data['Direction'] * cta_data['vol'] * cta_data['contract_multiplier']
            cta_diff_pnl = cta_data['diff_pnl'].sum().round(0)
        return quant_hold_value, quant_hold_pnl, quant_diff_pnl, cta_hold_value, cta_hold_pnl, cta_diff_pnl

    def _get_fut_assets(self, product: str) -> Tuple[float, float, float]:
        """获取期货静态权益、出金、入金"""
        fut_assets = self._read_product_files(product, 'fut_data', '_fut_assets')
        if fut_assets is None:
            return 0.0, 0.0, 0.0
        return (
            fut_assets['DynamicRights'].sum(),
            fut_assets['Withdraw'].sum(),
            fut_assets['Deposit'].sum()
        )

    def _get_opt_assets(self, product: str) -> Tuple[float, float]:
        """获取期权总资产和当日盈亏（当日-前日）"""
        opt_assets = self._read_product_files(product, 'opt_data', '_opt_assets')
        if opt_assets is None:
            return 0.0, 0.0

        total_assets = opt_assets['总资产'].sum()
        pnl = opt_assets['估算浮盈'].sum() if '估算浮盈' in opt_assets else 0.0

        # 获取前一日总资产
        pre_opt = self._read_product_files(product, 'opt_data', '_opt_assets', date=self.pre_date)
        if pre_opt is not None:
            pre_total = pre_opt['总资产'].sum()
            # 如果当日有浮盈，用当日总资产-浮盈推算昨日
            if pnl != 0 and pre_total == 0:
                pre_total = total_assets - pnl
        else:
            pre_total = total_assets - pnl

        return total_assets, total_assets - pre_total

    def _get_stk_assets(self, product: str) -> Tuple[float, float]:
        """获取股票账户净资产和可用资金"""
        assets_df = self._read_product_files(product, 'assets', '_assets')
        if assets_df is None:
            return 0.0, 0.0
        net_assets = assets_df['net_assets'].sum()
        ava_cash = assets_df['ava_cash'].sum()
        return net_assets, ava_cash

    def _get_stk_hold_value(self, product: str) -> float:
        """获取股票持仓市值"""
        pos_df = self._read_product_files(product, 'pos', '_pos')
        if pos_df is None or pos_df.empty:
            return 0.0

        # 合并收盘价
        pos_df['close'] = pos_df['code'].map(self.stk_close['close'])
        pos_df['hold_value'] = pos_df['close'] * pos_df['hold']
        return pos_df['hold_value'].sum()

    def _get_deposits_withdrawals(self, product: str) -> float:
        """获取出入金总额"""
        dw_path = os.path.join(self.standard_path, 'out_in', 'deposits_withdrawals.csv')
        if not os.path.exists(dw_path):
            return 0.0
        dw_df = pd.read_csv(dw_path, index_col=0)
        mask = (dw_df['date'].astype(str) == self.date) & (dw_df['product'] == product)
        return dw_df.loc[mask, 'deposite_withdrawwals'].sum() if mask.any() else 0.0

    def _get_pre_net(self, product: str) -> Tuple[float, float]:
        """获取前一日净资产（收盘价和结算价）"""
        pre_file = os.path.join(self.standard_path, 'net_data', self.pre_date,
                                f'{self.pre_date}_{product}_net_info.csv')
        if not os.path.exists(pre_file):
            return 0.0, 0.0, 0.0
        pre_df = pd.read_csv(pre_file)
        if pre_df.empty:
            return 0.0, 0.0, 0.0
        return pre_df.loc[0, '产品净资产(收盘价)'], pre_df.loc[0, '产品净资产(结算价)'], pre_df.loc[0, '期货户静态权益']

    def _get_pre_net_from_mail(self, product_chinese):
        # /home/zhanggh/DailyScripts/TradeData/standarddata/net_email/【基金净值】SAXM36_配邦恒升中性1号私募证券投资基金_2026-03-31.xlsx
        format_predate = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'
        if product_chinese == '配邦中圣1号':
            product_chinese = '配邦中圣'
        file_paths = self.match_file_v2(self.net_email_path, product_chinese, format_predate)
        if not file_paths:
            return
        data = pd.read_excel(file_paths)[:1].rename(columns={'日期': '净值日期', '资产份额净值(元)': '单位净值'})
        pre_net = data.loc[data['净值日期'] == format_predate, '单位净值']

        return pre_net

    def _cal_product_info(self, product: str, product_info: Dict) -> pd.DataFrame:
        """计算单个产品的净值信息"""
        product_chinese = product_info.get('product_chinese_name', '')
        product_type = product_info.get('product_type', '')
        benchmark = product_info.get('benchmark', 'hs300')

        # 获取各项数据
        stk_hold = self._get_stk_hold_value(product)
        stk_assets, stk_cash = self._get_stk_assets(product)
        fut_assets, fut_withdraw, fut_deposit = self._get_fut_assets(product)
        fut_diff_pnl, fut_hold_value, margin = self._get_fut_diff_pnl(product)
        opt_assets, opt_pnl = self._get_opt_assets(product)
        deposits = self._get_deposits_withdrawals(product)
        pre_net_close, pre_net_settle, pre_fut_assets_settle = self._get_pre_net(product)
        quant_order_pnl, quant_used_fee, cta_order_pnl, cta_used_fee = self._deal_fut_order(product)
        quant_hold_value, quant_hold_pnl, quant_diff_pnl, cta_hold_value, cta_hold_pnl, cta_diff_pnl = self._deal_fut_pos(product)

        # 计算净资产
        net_assets_close = stk_assets + fut_assets + opt_assets
        net_assets_settle = stk_assets + fut_assets + fut_diff_pnl + opt_assets

        # 计算净值增长率
        if pre_net_close != 0:
            net_growth_close = (net_assets_close - deposits) / pre_net_close - 1
        else:
            net_growth_close = np.nan

        if pre_net_settle != 0:
            net_growth_settle = (net_assets_settle - deposits) / pre_net_settle - 1
        else:
            net_growth_settle = np.nan

        benchmark_chg = self.index_df.loc[benchmark, 'chg'] if benchmark in self.index_df.index else np.nan

        # 从邮件获取产品准确昨日单位净值
        pre_net = self._get_pre_net_from_mail(product_chinese)

        # 产品信息
        data = pd.DataFrame([{
            '日期': self.date,
            '产品': product,
            '产品名': product_chinese,
            '出入金': deposits,
            '产品净资产(收盘价)': net_assets_close,
            '产品净资产(结算价)': net_assets_settle,
            '产品昨收净资产(收盘价)': pre_net_close,
            '产品昨收净资产(结算价)': pre_net_settle,
            '证券户净资产': stk_assets,
            '证券户持仓市值': stk_hold,
            '证券户可用现金': stk_cash,
            '期货户静态权益': fut_assets + fut_diff_pnl,
            '期货户出金': fut_withdraw,
            '期货户入金': fut_deposit,
            '期货户动态权益': fut_assets,
            '期货户名义市值': fut_hold_value,
            '期货户保证金': margin,
            '期货户风险度': margin / (fut_assets + fut_diff_pnl) if (fut_assets + fut_diff_pnl) != 0 else 0,
            '仓位': stk_hold / (stk_assets + fut_assets) if (stk_assets + fut_assets) != 0 else 0,
            '对冲比例': abs(stk_hold / fut_hold_value) if fut_hold_value != 0 else (np.nan if product_type == 'zx' else 0),
            '期权总资产': opt_assets,
            '基准': benchmark_chg,
            '净值增长': net_growth_close,
            '超额': net_growth_close - benchmark_chg if product_type != 'zx' else np.nan,
            '净值增长(结算价)': net_growth_settle,
            '超额(结算价)': net_growth_settle - benchmark_chg if product_type != 'zx' else np.nan,
        }])

        # 预估净值
        ros = pd.DataFrame([{
            '日期': self.date,
            '产品名': product_chinese,
            '昨日单位净值': pre_net,
            '净值增长': net_growth_settle,
            '当日预估净值': pre_net * (1 + net_growth_settle),
        }])

        # 期货数据
        fut_ros = pd.DataFrame([{
            '日期': self.date,
            '产品名': product_chinese,
            '昨日期货静态权益': pre_fut_assets_settle,
            '期货户静态权益': fut_assets + fut_diff_pnl,
            '期货户出入金': fut_deposit - fut_withdraw,
            '期货户收益': (fut_assets + fut_diff_pnl) - (pre_fut_assets_settle) + (fut_deposit - fut_withdraw),
            '期货户保证金': margin,
            '期货户风险度': margin / (fut_assets + fut_diff_pnl) if (fut_assets + fut_diff_pnl) != 0 else 0,
            'quant收益': quant_order_pnl - quant_used_fee + quant_hold_pnl,
            'cta收益': cta_order_pnl - cta_used_fee + cta_hold_pnl,
            'quant+cta收益': quant_order_pnl - quant_used_fee + quant_hold_pnl + cta_order_pnl - cta_used_fee + cta_hold_pnl,
        }])

        return data, ros, fut_ros

    def _deal_stk_order(self, acct, adj_fee_ratio=0.00015, t0_fee_ratio=0.0002):
        stk_order = self._read_product_files(acct, 'entrust', '_entrust')
        if stk_order is None:
            return 0, 0, 0, 0, 0, 0, 0
          
        stk_order = stk_order[stk_order['code'].isin(self.stk_close.index)]
        if stk_order.empty:
            return 0, 0, 0, 0, 0, 0, 0
           
        # stk_order = pd.merge(stk_order, self.stk_close, left_on='code', right_index=True, how='left')
        stk_order = pd.merge(stk_order, self.stk_close.reset_index(), on=['code', 'date'], how='left')

        stk_order['date'] = stk_order['date'].apply(lambda x: str(x))
        stk_order['filled_amount'] = stk_order['filled_vol'] * stk_order['filled_price']
        stk_order['fee'] = np.where(stk_order['algo'] == 't0',
                            np.where(stk_order['dir'] == 1, stk_order['filled_amount'] * t0_fee_ratio,
                                        stk_order['filled_amount'] * (t0_fee_ratio + 0.0005)),
                            np.where(stk_order['dir'] == 1, stk_order['filled_amount'] * adj_fee_ratio, stk_order['filled_amount'] * (adj_fee_ratio + 0.0005)))
        
        adj_data = stk_order[stk_order['algo'] == 'adj'].copy().reset_index(drop=True)
        adj_data['adj_benchmark'] = adj_data.apply(lambda row: row.get(f"{row['time_period']}_vwap"), axis=1)
        adj_data['benchmark_slipvwap'] = np.where(
            adj_data['filled_price'] != 0,
            (adj_data['filled_price'] /
            adj_data['adj_benchmark'] - 1) * 10000 * adj_data['dir'] * -1,
            0)
        adj_data['benchmark_slipvwap_amount'] = adj_data['benchmark_slipvwap'] * adj_data['filled_amount']
        # 调仓加权滑点
        stk_lng_adj_amount = adj_data['filled_amount'].sum().round(2)
        stk_lng_adj_slipvwap = (adj_data['benchmark_slipvwap_amount'].sum() / stk_lng_adj_amount).round(2) if stk_lng_adj_amount != 0 else 0

        # 调仓收益，策略收益
        adj_data['pnl'] = (adj_data['preclose'] - adj_data['filled_price']) * adj_data['filled_vol'] * adj_data['dir']
        adj_pnl = adj_data['pnl'].sum().round(2)
        adj_fee = adj_data['fee'].sum().round(2)
        stk_lng_adj_trade_act_pnl = adj_pnl - adj_fee

        # T0收益
        t0_data = stk_order[stk_order['algo'] == 't0'].copy().reset_index(drop=True)
        t0_data['cs_f'] = t0_data['filled_amount'] * t0_data['dir'] * -1
        tmp_t0_data = t0_data.copy()
        tmp_t0_data['vol'] = tmp_t0_data['dir'] * tmp_t0_data['filled_vol']
        grouped = tmp_t0_data[['code', 'vol']].groupby('code').sum().reset_index()
        t0_close_codes = grouped[grouped['vol'] == 0]['code'].tolist()
        t0_unclose_codes = grouped[grouped['vol'] != 0]['code'].tolist()
        t0_data = t0_data[t0_data.code.isin(t0_close_codes)].reset_index(drop=True)
        t0_pnl = t0_data['cs_f'].sum().round(2)
        t0_fee = t0_data['fee'].sum().round(2)
        stk_lng_t0_trade_act_amount = t0_data['filled_amount'].sum().round(2)
        stk_lng_t0_trade_act_pnl = t0_pnl - t0_fee
        stk_lng_t0_trade_expose_act_pnl = 0
        stk_sht_trade_pnl = 0
            
        return stk_lng_adj_amount, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl, stk_lng_t0_trade_act_amount, stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl

        pass

    def _deal_stk_pos(self, acct):
        stk_pos = self._read_product_files(acct, 'pos', '_pos')
        if stk_pos is None:
            return 0, 0, 0, 0
          
        stk_pos = stk_pos[stk_pos['code'].isin(self.stk_close.index)]
        if stk_pos.empty:
            return 0, 0, 0, 0
            
        stk_pos = pd.merge(stk_pos, self.stk_close, left_on='code', right_index=True, how='left')
        
        stk_pos['date'] = self.date
        stk_pos['hold_pnl'] = stk_pos['hold'] * (stk_pos['close'] - stk_pos['preclose'])
        stk_pos['hold_value'] = stk_pos['close'] * stk_pos['hold']
        stk_lng_hold_value = stk_pos['hold_value'].sum().round(0)
        stk_lng_hold_act_pnl = stk_pos['hold_pnl'].sum().round(0)
        stk_sht_hold_value = 0
        stk_sht_hold_act_pnl = 0

        return stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl

    def _cal_acct_info(self, product: str, product_chinese: str, benchmark: str, acct: str, acct_info: dict) -> pd.DataFrame:
        stk_lng_adj_amount, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl, stk_lng_t0_trade_act_amount, \
            stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl = self._deal_stk_order(acct)

        stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl = self._deal_stk_pos(acct)
        benchmarkg_chg = self.index_df.loc[benchmark, 'chg']
        stk_assets, stk_cash = self._get_stk_assets(acct)
        
        cangwei = stk_lng_hold_value / stk_assets if stk_assets != 0 else 0.0
        stk_lng_alpha_tot_pnl = stk_lng_hold_act_pnl + stk_lng_adj_trade_act_pnl
        stk_lng_alpha_tot_ret = stk_lng_alpha_tot_pnl / stk_lng_hold_value if stk_lng_hold_value != 0 else 0.0
        stk_sht_tot_pnl = stk_sht_hold_act_pnl + stk_sht_trade_pnl
        stk_lng_tot_pnl = stk_lng_alpha_tot_pnl + stk_lng_t0_trade_act_pnl
        stk_lng_tot_ret = stk_lng_tot_pnl / stk_lng_hold_value if stk_lng_hold_value != 0 else 0.0

        
        ros = pd.DataFrame([{
            '日期': self.date,
            '产品名': product_chinese,
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
            '空头总收益': stk_sht_tot_pnl,
            '证券户总收益': stk_lng_alpha_tot_pnl + stk_sht_tot_pnl,
            '多头收益率': stk_lng_alpha_tot_ret,
            '多头收益率(含T0)': stk_lng_tot_ret,
            '基准': benchmark,
            '基准收益率': benchmarkg_chg,
            '多头超额': (stk_lng_alpha_tot_ret - benchmarkg_chg) * cangwei,
            '多头超额(含T0)': (stk_lng_tot_ret - benchmarkg_chg) * cangwei,
        }])

        lng_ava_t0_amount = stk_lng_hold_value - stk_lng_adj_amount * 0.5
        t0_ros = pd.DataFrame([{
            '日期': self.date,
            '产品名': product_chinese,
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

        return ros, t0_ros


    # --------------------------- 主流程 ---------------------------
    def run_net(self) -> None:
        """运行计算流程"""
        all_data = []
        all_ros = []
        all_fut_ros = []

        for product, info in self.product_info.items():
            logger.info(f"计算产品: {product}")
            try:
                df, ros, fut_ros = self._cal_product_info(product, info)
                all_data.append(df)
                if product != 'PBZS1H':
                    all_ros.append(ros)
                    all_fut_ros.append(fut_ros)

                # 保存单个产品净值
                out_dir = os.path.join(self.standard_path, 'net_data', self.date)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'{self.date}_{product}_net_info.csv')
                df.to_csv(out_path, index=False)
                logger.info(f"保存 {product} 净值文件: {out_path}")
            except Exception as e:
                logger.error(f"计算产品 {product} 失败: {e}", exc_info=True)

        if not all_data:
            logger.warning("没有成功计算的产品")
            return

        # 合并所有产品信息
        combined = pd.concat(all_data, ignore_index=True)

        # 生成汇总表（用于展示）
        summary = combined[[
            '日期', '产品名', '出入金', '产品净资产(结算价)', '证券户净资产', '证券户持仓市值', '证券户可用现金',
            '仓位', '对冲比例', '基准', '净值增长', '超额', '净值增长(结算价)', '超额(结算价)'
        ]].copy()

        # 格式化输出
        summary['出入金'] = summary['出入金'].apply(lambda x: round(x / 10000, 2))
        summary['产品净资产(结算价)'] = summary['产品净资产(结算价)'].apply(lambda x: round(x / 10000, 2))
        summary['证券户净资产'] = summary['证券户净资产'].apply(lambda x: round(x / 10000, 2))
        summary['证券户持仓市值'] = summary['证券户持仓市值'].apply(lambda x: round(x / 10000, 2))
        summary['证券户可用现金'] = summary['证券户可用现金'].apply(lambda x: round(x / 10000, 2))
        summary['仓位'] = summary['仓位'].apply(lambda x: f"{x:.2%}")
        summary['对冲比例'] = summary['对冲比例'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else '--')
        summary['基准'] = summary['基准'].apply(lambda x: f"{x:.2%}")
        summary['净值增长'] = summary['净值增长'].apply(lambda x: f"{x:.2%}")
        summary['超额'] = summary['超额'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else '--')
        summary['净值增长(结算价)'] = summary['净值增长(结算价)'].apply(lambda x: f"{x:.2%}")
        summary['超额(结算价)'] = summary['超额(结算价)'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else '--')

        print(tabulate(summary, headers='keys', tablefmt='grid', showindex=False))

        # 保存汇总文件
        summary_path = os.path.join(self.standard_path, 'net_data', self.date,
                                    f'{self.date}_all_product_info.csv')
        combined.to_csv(summary_path, index=False)
        # logger.info(f"保存汇总文件: {summary_path}")
        
        self.run_lng_extra()

        # 期货户信息
        fut_info = pd.concat(all_fut_ros, ignore_index=True)
        fut_info['昨日期货静态权益'] = fut_info['昨日期货静态权益'].apply(lambda x: round(x / 10000, 2))
        fut_info['期货户静态权益'] = fut_info['期货户静态权益'].apply(lambda x: round(x / 10000, 2))
        fut_info['期货户出入金'] = fut_info['期货户出入金'].apply(lambda x: round(x / 10000, 2))
        fut_info['期货户收益'] = fut_info['期货户收益'].apply(lambda x: round(x / 10000, 2))
        fut_info['期货户保证金'] = fut_info['期货户保证金'].apply(lambda x: round(x / 10000, 2))
        fut_info['期货户风险度'] = fut_info['期货户风险度'].apply(lambda x: f"{x:.2%}")
        fut_info['quant收益'] = fut_info['quant收益'].apply(lambda x: round(x / 10000, 2))
        fut_info['cta收益'] = fut_info['cta收益'].apply(lambda x: round(x / 10000, 2))
        fut_info['quant+cta收益'] = fut_info['quant+cta收益'].apply(lambda x: round(x / 10000, 2))
        print(tabulate(fut_info, headers='keys', tablefmt='grid', showindex=False))

        # 预估净值
        res = pd.concat(all_ros, ignore_index=True)
        res['净值增长'] = res['净值增长'].apply(lambda x: f"{x:.2%}")
        res['当日预估净值'] = res['当日预估净值'].apply(lambda x: x.round(4))
        print(tabulate(res, headers='keys', tablefmt='grid', showindex=False))


    def run_lng_extra(self):
        
        all_data = []
        t0_res = []
        for product, product_info in self.product_info.items():
            product_chinese = product_info.get('product_chinese_name', '')
            benchmark = product_info.get('benchmark')
            # logger.info(f"计算产品: {product}, benchmark: {benchmark}")
            for acct, acct_info in self.product_info.get(product).get('acct_info').items():
                # logger.info(f"计算账户: {acct}")
                # logger.info(f"账户详情: {acct_info}")
                try:
                    df, t0_ros = self._cal_acct_info(product, product_chinese, benchmark, acct, acct_info)
                    all_data.append(df)
                    t0_res.append(t0_ros)
                    
                    # 保存单个产品净值
                    out_dir = os.path.join(self.standard_path, 'stk_data', self.date)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f'{self.date}_{acct}_stk_info.csv')
                    df.to_csv(out_path, index=False)
                    # logger.info(f"保存 {acct} 证券户信息: {out_path}")
                except Exception as e:
                    logger.error(f"计算账户 {acct} 失败: {e}", exc_info=True)

        if not all_data:
            logger.warning("没有成功计算的证券户信息")
            return
        
        ros = pd.concat(all_data, ignore_index=True)
        # 保存汇总文件
        summary_path = os.path.join(self.standard_path, 'stk_data', self.date,f'{self.date}_all_product_info.csv')
        ros.to_csv(summary_path, index=False)
        # logger.info(f"保存汇总文件: {summary_path}")

        res = ros[['日期', '产品名', '账户名', '证券户净资产', '多头持仓市值', '证券户总收益', '基准', '基准收益率', '多头超额', '多头超额(含T0)']].copy()
        res['证券户净资产'] = res['证券户净资产'].apply(lambda x: round(x/10000, 2))
        res['多头持仓市值'] = res['多头持仓市值'].apply(lambda x: round(x/10000, 2))
        res['证券户总收益'] = res['证券户总收益'].apply(lambda x: round(x/10000, 2))
        res['基准收益率'] = res['基准收益率'].apply(lambda x: f"{x:.2%}")
        res['多头超额'] = res['多头超额'].apply(lambda x: f"{x:.2%}")
        res['多头超额(含T0)'] = res['多头超额(含T0)'].apply(lambda x: f"{x:.2%}")
        print(tabulate(res, headers='keys', tablefmt='grid', showindex=False))

        # T0信息,调仓信息
        t0_df = pd.concat(t0_res, ignore_index=True)
        t0_df = t0_df[['日期', '产品名', '账户名', 't0_algo', '多头T0换手率(双边)', '多头T0收益', '多头T0收益率', 'adj_algo', '多头调仓滑点']]
        t0_df['多头T0收益'] = t0_df['多头T0收益'].apply(lambda x: round(x/10000, 2))
        t0_df['多头T0换手率(双边)'] = t0_df['多头T0换手率(双边)'].apply(lambda x: f"{x:.2%}")
        t0_df['多头T0收益率'] = t0_df['多头T0收益率'].apply(lambda x: f"{x:.2%}")
        t0_df['多头调仓滑点'] = t0_df['多头调仓滑点'].apply(lambda x: round(x, 2))
        print(tabulate(t0_df, headers='keys', tablefmt='grid', showindex=False))





def main():
    """主入口"""
    # 读取配置文件
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, 'split_system.yaml')
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 处理日期
    date = config.get('date', 'current')
    if date == 'current':
        date = dt.datetime.now().strftime('%Y%m%d')
    # date = '20260410'

    # 初始化 rqdatac
    rqdatac.init(
        config.get('rqdatac_user', '13601611030'),
        config.get('rqdatac_password', 'PB123456789')
    )

    # 运行计算
    calculator = ProductNetCalculator(date, config)
    calculator.run_net()
    # calculator.run_lng_extra()


if __name__ == '__main__':
    main()