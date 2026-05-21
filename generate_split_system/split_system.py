"""
split_system.py
读取目标权重文件 + 合并昨收持仓 → 生成交易拆分指令（调仓 / 建仓 / T0）
"""
import math
import os
import shutil
import sys
import yaml
import rqdatac
import pandas as pd
import numpy as np
import datetime as dt
import logging
import warnings
from pathlib import Path
from glob import glob

# 自动把 D:\code 加入 Python 搜索路径
sys.path.append(str(Path(__file__).parent.parent))

from ultis.format_client_algo import *
from ultis.email_manager import EmailManager

warnings.filterwarnings('ignore')

# 项目内模块
from config_loader import Config
from logger import setup_logger, get_logger


def match_file(path, key, date):
    """匹配指定路径下符合日期和关键词的最新文件"""
    pos_files = sorted(glob(f"{path}/{date}_{key}*_target.csv"))
    if len(pos_files) == 0:
        return None
    else:
        return pos_files[-1]


class SplitSystem(object):
    """持仓拆分系统：读取 target 权重 → 合并持仓 → 计算调仓量 → 生成交易指令"""

    def __init__(self, date: str, config: Config):
        self.date = date
        self.pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')

        self.data_path = config.get('data_path')
        self.raw_target_path = os.path.join(self.data_path, 'data', 'target')
        self.standard_path = os.path.join(self.data_path, 'data', 'standarddata')
        self.mo_order_path = os.path.join(self.data_path, 'data', 'mo_order')

        self.local_path = config.get('local_path')
        self.local_raw_target_path = os.path.join(self.local_path, 'data', 'target')
        self.local_standard_path = os.path.join(self.local_path, 'data', 'standarddata')
        self.local_mo_order_path = os.path.join(self.local_path, 'data', 'mo_order')

        self.lngexpose = config.get('lngexpose')
        self.product_info = config.get('product_info', {})
        self._logger = get_logger('split_system')

        self._logger.info(f'初始化完成 | 日期={self.date}  前交易日={self.pre_date}')

        # 获取行情（一次拉取，全局复用）
        self._logger.info('获取股票昨收行情...')
        self.stk_mkt_close = self.stk_mkt_close()
        self._logger.debug(f'  股票昨收: {len(self.stk_mkt_close)} 只')

        self._logger.info('获取期货昨收行情...')
        self.fut_close = self.get_fut_close()

    # ================================================================
    # 文件路径辅助：优先本地，不存在则使用远程
    # ================================================================
    def _resolve_read_path(self, local_path, remote_path):
        """读取文件时优先使用本地路径，不存在则退回到远程路径"""
        if os.path.exists(local_path):
            self._logger.debug(f'  读取本地文件: {local_path}')
            return local_path
        self._logger.debug(f'  本地文件不存在，读取远程: {remote_path}')
        return remote_path

    def stk_mkt_close(self):
        """获取股票昨收价格"""
        stk_mkt_codes = rqdatac.all_instruments(type='CS', market='cn', date=self.date)[
            'order_book_id'].unique().tolist()
        stk_mkt_df = rqdatac.get_price(
            stk_mkt_codes, start_date=self.pre_date, end_date=self.pre_date,
            frequency='1d', fields=None, adjust_type='pre', skip_suspended=False,
            market='cn', expect_df=True, time_slice=None,
        ).reset_index()
        stk_mkt_df = stk_mkt_df[['order_book_id', 'close']].rename(
            columns={'order_book_id': 'code'})
        stk_mkt_df['code'] = stk_mkt_df['code'].apply(
            lambda x: int(x.split('.')[0]))
        return stk_mkt_df

    def get_prepos(self, acct):
        """获取昨收持仓"""
        pos_paths_local = f'{self.local_standard_path}/pos/{self.pre_date}/{self.pre_date}_{acct}_pos.csv'
        pos_paths_remote = f'{self.standard_path}/pos/{self.pre_date}/{self.pre_date}_{acct}_pos.csv'
        pos_paths = self._resolve_read_path(pos_paths_local, pos_paths_remote)
        if os.path.exists(pos_paths):
            pos = pd.read_csv(pos_paths, index_col=0)
            self._logger.debug(f'  昨收持仓 {acct}: {len(pos)} 条')
            return pos
        else:
            self._logger.warning(f'  昨收持仓文件缺失: {os.path.basename(pos_paths)}')
            return pd.DataFrame(columns=['code', 'hold'])

    def get_raw_target(self, uid):
        """获取原始 target 文件"""
        target_path_local = f'{self.local_raw_target_path}/{self.date}'
        target_paths_local = match_file(target_path_local, uid, self.date)
        target_path_remote = f'{self.raw_target_path}/{self.date}'
        target_paths_remote = match_file(target_path_remote, uid, self.date)
        target_paths = target_paths_local if target_paths_local is not None else target_paths_remote
        if target_paths is None:
            raise FileNotFoundError(
                f'未找到 target 文件: {target_path}/{self.date}_{uid}_*_target.csv')
        self._logger.debug(f'  读取 target: {os.path.basename(target_paths)}')
        target = pd.read_csv(target_paths)
        target = target.rename(columns={'ticker': 'code', 'weight': 'w'})
        target['code'] = target['code'].apply(lambda x: int(str(x)[:6]))
        return target

    # ---- 取整逻辑 ----
    def _min_trade_unit(self, code):
        """根据股票代码判断最小交易单位（科创板200股，其他100股）"""
        return 200 if code >= 688000 else 100

    def _round_lot1(self, code, hold, close, vol):
        """将调整量按最小交易单位取整（旧版）"""
        if code >= 688000:
            if vol > 0:
                tvol = max(200, vol) if vol >= 100 else 0
            else:
                tvol = (max(min(-200, vol), -hold) if vol <= -100 else (-hold if vol <= -hold else 0))
        else:
            if vol > 0:
                tvol = round(vol / 100, 0) * 100 if vol >= 50 else 0
            else:
                vol = round(vol / 100, 0) * 100
                tvol = (max(min(-100, vol), -hold) if vol <= -50 else (-hold if vol <= -hold else 0))
        return int(tvol)

    def _round_lot2(self, code, hold, close, vol):
        """将调整量按最小交易单位取整（新版）"""
        if code >= 688000:
            if vol > 0:
                tvol = max(200, vol) if vol >= 100 else (200 if close >= 100 else 0)
            else:
                tvol = (max(min(-200, vol), -hold) if vol <= -100 else (-hold if vol <= -hold else 0))
        else:
            if vol > 0:
                tvol = (
                    round(vol / 100, 0) * 100
                    if vol >= 50
                    else (100 if close >= 100 else 0)
                )
            else:
                vol = round(vol / 100, 0) * 100
                tvol = (max(min(-100, vol), -hold) if vol <= -50 else (-hold if vol <= -hold else 0))
        return int(tvol)

    def _round_lot(self, code, hold, close, vol, version='new'):
        if version == 'new':
            return self._round_lot2(code, hold, close, vol)
        else:
            return self._round_lot1(code, hold, close, vol)

    # ---- 偏差修正 ----
    def _adjust_for_deviation(self, data, base_value, target_value, round_lot):
        """目标市值与基准市值偏差超 0.5% 时微调"""
        deviation = target_value - base_value
        if abs(deviation) / base_value <= 0.005:
            return data, target_value, 0

        need_value = abs(deviation)
        direction = -1 if deviation > 0 else 1

        candidates = data[(data['w'] > 0)].copy()
        candidates = candidates.sort_values('w', ascending=False)
        if candidates.empty:
            return data, target_value, 0

        candidates['adjust_vol'] = (need_value + target_value * 0.02) * candidates['w'] / candidates['close']
        candidates['adjust_vol1'] = candidates.apply(
            lambda row: self._round_lot(row['code'], 0, 10, row['adjust_vol'], version=round_lot), axis=1)
        candidates['adjust_val'] = candidates['adjust_vol1'] * candidates['close']
        candidates['adjust_val_cumsum'] = candidates['adjust_val'].cumsum()
        candidates = candidates[candidates['adjust_val_cumsum'] <= need_value].copy().reset_index()

        adjustments = []
        for _, row in candidates.iterrows():
            code = int(row['code'])
            target1 = row['target1']
            if direction == 1:
                adjustments.append((code, int(row['adjust_vol1'])))
            else:
                sell_vol = min(target1, int(row['adjust_vol1']))
                adjustments.append((code, -sell_vol))

        if not adjustments:
            return data, target_value, 0

        for code, delta in adjustments:
            mask = data['code'] == code
            data.loc[mask, 'target1'] += delta
            data.loc[mask, 'adj1'] += delta

        data['adj1'] = data.apply(
            lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['adj1'], version=round_lot), axis=1)
        data['target1'] = data['hold'] + data['adj1']

        new_target_value = (data['target1'] * data['close']).sum()
        added_value = new_target_value - target_value
        direction_label = '买入' if direction == 1 else '卖出'
        self._logger.info(
            f'  偏差修正: {direction_label} {added_value:.0f}元, 目标市值 {new_target_value:.0f}')
        return data, new_target_value, added_value

    # ---- 合并持仓与目标权重 ----
    def combine_pos_target(self, acct, is_clear, round_lot, uid, adjust_value, pre_reduce_value, pre_build_value):
        """合并昨收持仓和 target，计算调整量"""
        prepos = self.get_prepos(acct)
        if not is_clear:
            raw_target = self.get_raw_target(uid)
            merge_df = pd.merge(prepos, raw_target, on='code', how='outer')
        else:
            merge_df = prepos.copy()
            merge_df['w'] = 0

        merge_df = pd.merge(merge_df, self.stk_mkt_close, on='code', how='left').fillna(0)

        prepos_value = (merge_df['hold'] * merge_df['close']).sum()
        base_value = prepos_value + adjust_value

        self._logger.info(f'--昨收持仓市值：{prepos_value}')
        self._logger.info(f'--手动调整市值：{adjust_value}')
        self._logger.info(f'--理论目标市值：{base_value}')

        # 权重 → 目标股数
        merge_df['per_value'] = base_value * merge_df['w']
        merge_df['target'] = (merge_df['per_value'] / merge_df['close']).fillna(0).round(0)
        merge_df['adj'] = merge_df['target'] - merge_df['hold']

        merge_df['adj1'] = merge_df.apply(
            lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['adj'], version=round_lot), axis=1)
        merge_df['target1'] = merge_df['hold'] + merge_df['adj1']

        target_value = (merge_df['target1'] * merge_df['close']).sum()
        self._logger.info(f'--目标持仓市值（取整后）：{target_value}')

        merge_df, target_value, added = self._adjust_for_deviation(merge_df, base_value, target_value, round_lot)

        # 预留减仓（暂为0，后续可配置）
        merge_df['reduce_vol'] = 0
        merge_df['reduce_vol1'] = 0

        # 建仓单
        self._logger.info(f'--预备建仓市值：{pre_build_value}')
        merge_df['per_build_value'] = pre_build_value * merge_df['w']
        merge_df['build_vol'] = (merge_df['per_build_value'] / merge_df['close']).fillna(0).round(0)
        if pre_build_value == 0:
            merge_df['build_vol1'] = 0
        else:
            merge_df['build_vol1'] = merge_df.apply(
                lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['build_vol'], version=round_lot), axis=1)
        build_value = (merge_df['build_vol1'] * merge_df['close']).sum()
        self._logger.info(f'--建仓单市值：{build_value}')

        # 剩余可 T0 头寸
        merge_df['left_vol'] = np.where(
            merge_df['adj1'] >= 0,
            merge_df['hold'] + merge_df['reduce_vol'],
            merge_df['hold'] + merge_df['adj1'] + merge_df['reduce_vol'],
        )

        return merge_df

    def split_cangdan(self, data):
        """将调仓、减仓、建仓、T0单分开"""
        adj_df = data[['code', 'adj1', 'close']]
        adj_df = adj_df[adj_df['adj1'] != 0].reset_index(drop=True)
        reduce_df = data[['code', 'reduce_vol1', 'close']]
        reduce_df = reduce_df[reduce_df['reduce_vol1'] != 0].reset_index(drop=True)
        build_df = data[['code', 'build_vol1', 'close']]
        build_df = build_df[build_df['build_vol1'] != 0].reset_index(drop=True)
        t0_df = data[['code', 'left_vol', 'close']]
        t0_df = t0_df[t0_df['left_vol'] != 0].reset_index(drop=True)
        return adj_df, reduce_df, build_df, t0_df

    def format_code(self, code):
        """将6位数字代码转换为带后缀的格式"""
        code = int(str(code)[:6])
        code = str(code).zfill(6)
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
        return code

    def get_fut_close(self):
        """获取期货昨收行情"""
        fut_mkt_df = rqdatac.all_instruments(type='Future', market='cn', date=self.date)
        fut_mkt_df = fut_mkt_df[
            fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))
        ].reset_index(drop=True)
        fut_mkt_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(
            columns={'order_book_id': 'code'})
        fut_mkt_codes = fut_mkt_df['code'].tolist()
        fut_mkt = rqdatac.get_price(
            fut_mkt_codes, start_date=self.pre_date, end_date=self.pre_date,
            frequency='1d', fields=None, adjust_type='pre', skip_suspended=False,
            market='cn', expect_df=True, time_slice=None,
        ).reset_index()
        fut_mkt = fut_mkt[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(
            columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
        fut_mkt_df = pd.merge(fut_mkt_df, fut_mkt, on='code', how='left')
        return fut_mkt_df

    def process_split(self, acct, acct_info):
        """处理单个账户的拆分，返回汇总 dict"""
        uid = acct_info.get('target_uid')
        adjust_value = acct_info.get('adjust_value', 0)
        pre_reduce_value = acct_info.get('pre_reduce_value', 0)
        pre_build_value = acct_info.get('pre_build_value', 0)
        rzrq = acct_info.get('rzrq', False)
        is_clear = acct_info.get('is_clear', False)
        is_rzrq = True if rzrq else False
        round_lot = acct_info.get('round_lot', 'new')

        data = self.combine_pos_target(acct, is_clear, round_lot, uid, adjust_value, pre_reduce_value, pre_build_value)

        pre_hold_value = (data['hold'] * data['close']).sum()
        adj_df, reduce_df, build_df, t0_df = self.split_cangdan(data)

        summary = {'acct': acct, 'adj_buy': 0, 'adj_sell': 0, 'adj_count': 0,
                    'adj_ratio': 0, 'build_count': 0, 't0_count': 0, 't0_value': 0}

        # ---- 调仓单 ----
        if acct_info.get('is_adj'):
            adj_df['adj_value'] = adj_df['adj1'] * adj_df['close']
            adj_buy = adj_df[adj_df['adj1'] >= 0]['adj_value'].sum()
            adj_sell = adj_df[adj_df['adj1'] < 0]['adj_value'].sum()
            adj_ratio = (adj_buy - adj_sell) * 0.5 / pre_hold_value if pre_hold_value != 0 else 0

            summary['adj_buy'] = adj_buy
            summary['adj_sell'] = adj_sell
            summary['adj_count'] = len(adj_df)
            summary['adj_ratio'] = adj_ratio

            self._logger.info(f'--调仓买入金额：{adj_buy}')
            self._logger.info(f'--调仓卖出金额：{adj_sell}')

            format_adj_algo(
                self.mo_order_path, adj_df, self.date,
                acct_info.get('adj_client'), acct_info.get('adj_acct'), acct_info.get('adj_algo'),
                acct, acct_info.get('adj_algo_starttime'), acct_info.get('adj_algo_endtime'),
                rzrq=is_rzrq,
            )

        # ---- 建仓单 ----
        if pre_build_value != 0 and not build_df.empty:
            build_df = build_df.rename(columns={'build_vol1': 'adj1'})
            summary['build_count'] = len(build_df)
            format_adj_algo(
                self.mo_order_path, build_df, self.date,
                acct_info.get('adj_client'), acct_info.get('adj_acct'), acct_info.get('adj_algo'),
                acct, acct_info.get('adj_algo_starttime'), acct_info.get('adj_algo_endtime'),
                rzrq=is_rzrq, algo_type='build',
            )

        # ---- T0 单 ----
        if acct_info.get('is_t0') and not t0_df.empty:
            t0_df['t0_value'] = t0_df['left_vol'] * t0_df['close']
            t0_amount = t0_df['t0_value'].sum()
            summary['t0_count'] = len(t0_df)
            summary['t0_value'] = t0_amount

            self._logger.info(f'--可T0股票数量：{len(t0_df)}')
            self._logger.info(f'--可T0股票市值：{t0_amount}')
            format_t0_algo(
                self.mo_order_path, t0_df, self.date,
                acct_info.get('t0_client'), acct_info.get('t0_acct'), acct_info.get('t0_algo'),
                acct, acct_info.get('t0_algo_starttime'), acct_info.get('t0_algo_endtime'),
            )

        # 同步订单文件到本地
        try:
            src_mo_dir = os.path.join(self.mo_order_path, self.date)
            dst_mo_dir = os.path.join(self.local_mo_order_path, self.date)
            if os.path.exists(src_mo_dir):
                shutil.copytree(src_mo_dir, dst_mo_dir, dirs_exist_ok=True)
                self._logger.info(f'  订单文件已同步到本地: {dst_mo_dir}')
        except Exception as copy_err:
            self._logger.warning(f'  订单文件同步到本地失败: {copy_err}')

        return summary

    def main(self):
        """主流程：遍历所有产品和账户"""
        products = list(self.product_info.items())
        total_products = len(products)
        all_results = []

        for pidx, (product, product_info) in enumerate(products, 1):
            product_chinese_name = product_info.get('product_chinese_name', '')
            accts = list(product_info.get('acct_info', {}).items())
            total_accts = len(accts)

            self._logger.info(f'{"─" * 55}')
            self._logger.info(f'[{pidx}/{total_products}] {product} {product_chinese_name}  ({total_accts}个账户)')
            self._logger.info(f'产品：{product} {product_chinese_name}')

            for aidx, (acct, acct_info) in enumerate(accts, 1):
                acct_chinese_name = acct_info.get('acct_chinese_name', '')
                uid = acct_info.get('target_uid', '?')
                adj_client = acct_info.get('adj_client', '')

                # ---- 外部权重邮件下载 ----
                download_cfg = acct_info.get('download_target_email')
                if download_cfg:
                    target_dir = f'{self.raw_target_path}/{self.date}'
                    rename_to = download_cfg.get('rename_to', '').replace('{date}', self.date)
                    file_pattern = download_cfg.get('file_pattern', '').replace('{date}', self.date)
                    expected_file = f'{target_dir}/{rename_to}'

                    if not os.path.exists(expected_file):
                        keywords = [
                            kw.replace('{date}', self.date) if '{date}' in kw else kw
                            for kw in download_cfg.get('keywords', [])
                        ]
                        self._logger.info(f'  下载外部权重: {keywords}')
                        manager = EmailManager()
                        try:
                            print(keywords)
                            manager.download_attachments_by_keyword(
                                keywords, save_dir=target_dir,
                                file_extensions=download_cfg.get('file_extensions', ['.csv']), limit=50)
                            manager.logout()
                            src = f'{target_dir}/{file_pattern}'
                            if os.path.exists(src):
                                os.rename(src, expected_file)
                                self._logger.info(f'  权重文件已保存: {rename_to}')
                                # 同步到本地
                                local_target_date = os.path.join(self.local_raw_target_path, self.date)
                                os.makedirs(local_target_date, exist_ok=True)
                                shutil.copy2(expected_file, os.path.join(local_target_date, rename_to))
                                self._logger.info(f'  已同步到本地: {local_target_date}')
                            else:

                                self._logger.warning(f'  下载后未找到文件: {file_pattern}')
                        except Exception as e:
                            self._logger.warning(f'  下载失败: {e}')

                # 处理
                self._logger.info(f'--账户：{acct} {acct_chinese_name}')
                try:
                    summary = self.process_split(acct, acct_info)

                    # 一行摘要
                    parts = [f'{acct}({uid})']
                    if summary['adj_count']:
                        parts.append(
                            f'调仓{summary["adj_count"]}只 '
                            f'买{summary["adj_buy"]/1e4:.0f}w 卖{abs(summary["adj_sell"])/1e4:.0f}w '
                            f'换手{summary["adj_ratio"]:.2%}')
                    if summary['build_count']:
                        parts.append(f'建仓{summary["build_count"]}只')
                    if summary['t0_count']:
                        parts.append(f'T0 {summary["t0_count"]}只({summary["t0_value"]/1e4:.0f}w)')
                    self._logger.info(f'  [{aidx}/{total_accts}] {" | ".join(parts)}')

                    all_results.append((acct, True, summary))
                except Exception as e:
                    self._logger.error(f'  [{aidx}/{total_accts}] {acct} ✗ 失败: {e}')
                    all_results.append((acct, False, None))

        # ---- 汇总 ----
        ok = sum(1 for _, s, _ in all_results if s)
        fail = len(all_results) - ok
        total_adj_buy = sum(r['adj_buy'] for _, s, r in all_results if s and r)
        total_adj_sell = sum(r['adj_sell'] for _, s, r in all_results if s and r)

        self._logger.info(f'{"=" * 55}')
        self._logger.info(f'处理汇总: 成功 {ok}/{len(all_results)}  失败 {fail}/{len(all_results)}')
        self._logger.info(f'  总调仓买入: {total_adj_buy/1e4:.0f}w  总调仓卖出: {abs(total_adj_sell)/1e4:.0f}w')
        for acct, status, summary in all_results:
            mark = '✓' if status else '✗'
            if summary:
                self._logger.info(
                    f'  [{mark}] {acct}  |  '
                    f'调仓{summary["adj_count"]}只  '
                    f'买{summary["adj_buy"]/1e4:.0f}w 卖{abs(summary["adj_sell"])/1e4:.0f}w  '
                    f'换手{summary["adj_ratio"]:.2%}  |  '
                    f'建仓{summary["build_count"]}只  T0 {summary["t0_count"]}只')
            else:
                self._logger.info(f'  [{mark}] {acct}  |  失败')
        self._logger.info(f'{"=" * 55}')


# ============================= 入口 =============================
if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))

    config = Config(f'{root_path}/split_system.yaml')

    log_cfg = config.get('logging', {})
    logger = setup_logger(
        name='split_system',
        log_dir=log_cfg.get('log_dir'),
        level=log_cfg.get('level', 'INFO'),
        max_bytes=log_cfg.get('max_bytes', 10 * 1024 * 1024),
        backup_count=log_cfg.get('backup_count', 30),
    )

    rq_cfg = config.get('rqdatac', {})
    rqdatac.init(
        username=rq_cfg.get('username', 'license'),
        password=rq_cfg.get('password', ''),
        use_pool=rq_cfg.get('use_pool', True),
        max_pool_size=rq_cfg.get('max_pool_size', 8),
    )
    logger.info('rqdatac 初始化完成')

    date = sys.argv[1] if len(sys.argv) > 1 else config.get('date')
    logger.info(f'split_system 启动 | 日期={date}')

    SS = SplitSystem(date, config)
    SS.main()

    logger.info('split_system 全部完成')
