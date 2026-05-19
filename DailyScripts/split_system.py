import math
import os
from sys import version
import yaml
import rqdatac
import pandas as pd
import numpy as np
import datetime as dt
from glob import glob
from ultis.format_client_algo import *
from ultis.log_msg import *
import warnings
warnings.filterwarnings('ignore')
from ultis.email_manager import EmailManager

# 初始化 RiceQuant
# rqdatac.init(13601611030, 'PB123456789', use_pool=True, max_pool_size=8)
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)


# 获取当前脚本的文件名
script_name = os.path.basename(__file__)


def match_file(path, key, date):
    """匹配指定路径下符合日期和关键词的最新文件"""
    pos_files = sorted(glob(f"{path}/{date}_{key}*_target.csv"))
    if len(pos_files) == 0:
        return None
    else:
        return pos_files[-1]


class SplitSystem(object):
    def __init__(self, date, config):
        self.date = date
        self.pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
        self.raw_target_path = config.get('raw_target_path')
        self.standard_path = config.get('standard_path')
        self.out_path = config.get('out_path')
        self.lngexpose = config.get('lngexpose')
        self.product_info = config.get('product_info')
        self.stk_mkt_close = self.stk_mkt_close()
        self.fut_close = self.get_fut_close()

    def stk_mkt_close(self):
        """获取股票昨收价格"""
        stk_mkt_codes = rqdatac.all_instruments(type='CS', market='cn', date=self.date)[
            'order_book_id'].unique().tolist()
        stk_mkt_df = rqdatac.get_price(stk_mkt_codes, start_date=self.pre_date, end_date=self.pre_date,
                                       frequency='1d', fields=None, adjust_type='pre', skip_suspended=False,
                                       market='cn', expect_df=True, time_slice=None).reset_index()
        stk_mkt_df = stk_mkt_df[['order_book_id', 'close']].rename(
            columns={'order_book_id': 'code'})
        stk_mkt_df['code'] = stk_mkt_df['code'].apply(
            lambda x: int(x.split('.')[0]))
        return stk_mkt_df

    def get_prepos(self, acct):
        """获取昨收持仓"""
        pos_paths = f'{self.standard_path}/pos/{self.pre_date}/{self.pre_date}_{acct}_pos.csv'
        if os.path.exists(pos_paths):
            pos = pd.read_csv(pos_paths, index_col=0)
            return pos
        else:
            return pd.DataFrame(columns=['code', 'hold'])

    def get_raw_target(self, uid):
        """获取原始target文件"""

        target_path = f'{self.raw_target_path}/{self.date}'
        target_paths = match_file(target_path, uid, self.date)
        target = pd.read_csv(target_paths) # , index_col=0
        target = target.rename(columns={'ticker': 'code', 'weight': 'w'})
        target['code'] = target['code'].apply(lambda x: int(str(x)[:6]))
        return target

    def _min_trade_unit(self, code):
        """根据股票代码判断最小交易单位（科创板200股，其他100股）"""
        return 200 if code >= 688000 else 100

    def _round_lot1(self, code, hold, close, vol):
        """将调整量按最小交易单位取整"""
        if code >= 688000:  # 科创板
            if vol > 0:
                tvol = max(200, vol) if vol >= 100 else 0
            else:
                tvol = (max(min(-200, vol), -hold) if vol <= -100 else (-hold if vol <= -hold else 0))
        else:  # 非科创板
            if vol > 0:
                tvol = round(vol / 100, 0) * 100 if vol >= 50 else 0
            else:
                vol = round(vol / 100, 0) * 100
                tvol = (max(min(-100, vol), -hold) if vol <= -50 else (-hold if vol <= -hold else 0))
        return int(tvol)
    
    def _round_lot2(self, code, hold, close, vol):
        """将调整量按最小交易单位取整"""
        if code >= 688000:  # 科创板
            if vol > 0:
                tvol = max(200, vol) if vol >= 100 else (200 if close >= 100 else 0)
            else:
                tvol = (max(min(-200, vol), -hold) if vol <= -100 else (-hold if vol <= -hold else 0))
        else:  # 非科创板
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


    def _adjust_for_deviation(self, data, base_value, target_value, round_lot):
        """
        当目标市值与基准市值偏差超过0.5%时，根据权重理论值对 target1 进行微调，
        使调整后的市值尽可能接近基准市值。
        """
        deviation = target_value - base_value
        if abs(deviation) / base_value <= 0.005:
            return data, target_value, 0

        need_value = abs(deviation)
        direction = -1 if deviation > 0 else 1  # -1: 需要卖出（减少市值），1: 需要买入（增加市值）

        # 筛选候选股票：保留有权重的股票（与原逻辑一致）
        candidates = data[(data['w'] > 0)].copy()
        candidates = candidates.sort_values('w', ascending=False)
        if candidates.empty:
            return data, target_value, 0


        candidates['adjust_vol'] = (need_value + target_value * 0.02) * candidates['w'] / candidates['close']
        candidates['adjust_vol1'] = candidates.apply(lambda row: self._round_lot(row['code'], 0, 10, row['adjust_vol'], version=round_lot), axis=1)
        candidates['adjust_val'] = candidates['adjust_vol1'] * candidates['close']
        candidates['adjust_val_cumsum'] = candidates['adjust_val'].cumsum()
        candidates = candidates[candidates['adjust_val_cumsum'] <= need_value].copy().reset_index()
        
        adjustments = []       # 记录 (code, delta_vol)
        for _, row in candidates.iterrows():
            code = int(row['code'])
            target1 = row['target1']
            if direction == 1:  # 买入
                buy_vol = int(row['adjust_vol1']) 
                adjustments.append((code, buy_vol))
            else:               # 卖出
                sell_vol = int(row['adjust_vol1'])
                sell_vol = min(target1, sell_vol)
                adjustments.append((code, -sell_vol))

        if not adjustments:
            return data, target_value, 0

        # 应用调整
        for code, delta in adjustments:
            mask = data['code'] == code
            data.loc[mask, 'target1'] += delta
            data.loc[mask, 'adj1'] += delta

        data['adj1'] = data.apply(lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['adj1'], version=round_lot), axis=1)
        data['target1'] = data['hold'] + data['adj1']
        
        new_target_value = (data['target1'] * data['close']).sum()
        added_value = new_target_value - target_value
        log_message(f'--- 偏差修正完成，调整方向 {"买入" if direction==1 else "卖出"}，调整市值 {added_value:.2f} 元，新目标市值 {new_target_value:.2f} 元',
                    script_name=script_name)
        return data, new_target_value, added_value

    def combine_pos_target(self, acct, is_clear, round_lot, uid, adjust_value, pre_reduce_value, pre_build_value):
        """合并昨收持仓和target，计算调整量，并进行偏差修正"""
        prepos = self.get_prepos(acct)
        if not is_clear:
            raw_target = self.get_raw_target(uid)
            merge_df = pd.merge(prepos, raw_target, on='code', how='outer')
        else:
            raw_target = pd.DataFrame(columns=['code', 'w'])
            merge_df = prepos.copy()
            merge_df['w'] = 0

        merge_df = pd.merge(merge_df, self.stk_mkt_close, on='code', how='left').fillna(0)

        # 基准市值 = 昨收持仓市值 + 手动调整市值
        prepos_value = (merge_df['hold'] * merge_df['close']).sum()
        base_value = prepos_value + adjust_value
        log_message(f'--昨收持仓市值：{prepos_value}', script_name=script_name)
        log_message(f'--手动调整市值：{adjust_value}', script_name=script_name)
        log_message(f'--理论目标市值：{base_value}', script_name=script_name)

        # 按权重计算目标股数（未取整）
        merge_df['per_value'] = base_value * merge_df['w']
        merge_df['target'] = (merge_df['per_value'] / merge_df['close']).fillna(0).round(0)
        merge_df['adj'] = merge_df['target'] - merge_df['hold']

        # 对调整量进行取整
        merge_df['adj1'] = merge_df.apply(
            lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['adj'], version=round_lot), axis=1
        )
        merge_df['target1'] = merge_df['hold'] + merge_df['adj1']

        # 初始目标市值
        target_value = (merge_df['target1'] * merge_df['close']).sum()
        log_message(f'--目标持仓市值（取整后）：{target_value}', script_name=script_name)

        # 偏差修正：若偏差超过0.5%则进行微调
        merge_df, target_value, added = self._adjust_for_deviation(merge_df, base_value, target_value, round_lot)

        # 预留减仓（暂为0）
        merge_df['reduce_vol'] = 0
        merge_df['reduce_vol1'] = 0

        # 建仓单处理（独立于调仓）
        log_message(f'--预备建仓市值：{pre_build_value}', script_name=script_name)
        merge_df['per_build_value'] = pre_build_value * merge_df['w']
        merge_df['build_vol'] = (merge_df['per_build_value'] / merge_df['close']).fillna(0).round(0)
        if pre_build_value == 0:
            merge_df['build_vol1'] = 0
        else:
            merge_df['build_vol1'] = merge_df.apply(
                lambda row: self._round_lot(row['code'], row['hold'], row['close'], row['build_vol'], version=round_lot), axis=1
            )
        build_value = (merge_df['build_vol1'] * merge_df['close']).sum()
        log_message(f'--建仓单市值：{build_value}', script_name=script_name)

        # 计算剩余可T0头寸
        merge_df['left_vol'] = np.where(
            merge_df['adj1'] >= 0,
            merge_df['hold'] + merge_df['reduce_vol'],
            merge_df['hold'] + merge_df['adj1'] + merge_df['reduce_vol']
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
        fut_mkt_df = fut_mkt_df[fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))].reset_index(drop=True)
        fut_mkt_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(
            columns={'order_book_id': 'code'})
        fut_mkt_codes = fut_mkt_df['code'].tolist()
        fut_mkt = rqdatac.get_price(fut_mkt_codes, start_date=self.pre_date, end_date=self.pre_date,
                                    frequency='1d', fields=None, adjust_type='pre', skip_suspended=False,
                                    market='cn', expect_df=True, time_slice=None).reset_index()
        fut_mkt = fut_mkt[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(
            columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
        fut_mkt_df = pd.merge(fut_mkt_df, fut_mkt, on='code', how='left')
        return fut_mkt_df

    def process_split(self, acct, acct_info):
        """处理单个账户的拆分"""
        uid = acct_info.get('target_uid')
        adjust_value = acct_info.get('adjust_value', 0)
        pre_reduce_value = acct_info.get('pre_reduce_value', 0)
        pre_build_value = acct_info.get('pre_build_value', 0)
        rzrq = acct_info.get('rzrq', False)
        is_clear = acct_info.get('is_clear', False)
        is_rzrq = True if rzrq else False
        round_lot = acct_info.get('round_lot', 'new')
        print(round_lot)

        data = self.combine_pos_target(acct, is_clear, round_lot, uid, adjust_value, pre_reduce_value, pre_build_value)
        # df_target = data[['code', '']]

        pre_hold_value = (data['hold'] * data['close']).sum()
        adj_df, reduce_df, build_df, t0_df = self.split_cangdan(data)

        # 调仓单
        if acct_info.get('is_adj'):
            adj_df['adj_value'] = adj_df['adj1'] * adj_df['close']
            adj_buy = adj_df[adj_df['adj1'] >= 0]['adj_value'].sum()
            adj_sell = adj_df[adj_df['adj1'] < 0]['adj_value'].sum()
            log_message(f'--调仓买入金额：{adj_buy}', script_name=script_name)
            log_message(f'--调仓卖出金额：{adj_sell}', script_name=script_name)
            adj_ratio = (adj_buy - adj_sell) * 0.5 / pre_hold_value if pre_hold_value != 0 else 0
            log_message(f'--调仓换手率：{format(adj_ratio, ".2%")}', script_name=script_name)
            format_adj_algo(
                self.out_path, adj_df, self.date,
                acct_info.get('adj_client'), acct_info.get('adj_acct'), acct_info.get('adj_algo'),
                acct, acct_info.get('adj_algo_starttime'), acct_info.get('adj_algo_endtime'),
                rzrq=is_rzrq
            )

        # 建仓单
        if pre_build_value != 0 and not build_df.empty:
            build_df = build_df.rename(columns={'build_vol1': 'adj1'})
            format_adj_algo(
                self.out_path, build_df, self.date,
                acct_info.get('adj_client'), acct_info.get('adj_acct'), acct_info.get('adj_algo'),
                acct, acct_info.get('adj_algo_starttime'), acct_info.get('adj_algo_endtime'),
                rzrq=is_rzrq, algo_type='build'
            )

        # T0单
        if acct_info.get('is_t0') and not t0_df.empty:
            t0_df['t0_value'] = t0_df['left_vol'] * t0_df['close']
            t0_amount = t0_df['t0_value'].sum()
            log_message(f'--可T0股票数量：{len(t0_df)}', script_name=script_name)
            log_message(f'--可T0股票市值：{t0_amount}', script_name=script_name)
            format_t0_algo(
                self.out_path, t0_df, self.date,
                acct_info.get('t0_client'), acct_info.get('t0_acct'), acct_info.get('t0_algo'),
                acct, acct_info.get('t0_algo_starttime'), acct_info.get('t0_algo_endtime')
            )

    def main(self):
        """主流程：遍历所有产品和账户"""
        for product, product_info in self.product_info.items():
            product_chinese_name = product_info.get('product_chinese_name', '')
            print(' ')
            log_message(f'产品：{product} {product_chinese_name}', script_name=script_name)

            for acct, acct_info in product_info.get('acct_info', {}).items():
                if acct == 'gtht_PBHSZX1H':
                    if not os.path.exists(f'{self.raw_target_path}/{self.date}/{self.date}_CZQZX300_000300.XSHG_000906.XSHG_target.csv'):
                        manager = EmailManager()
                        try:
                            manager.download_attachments_by_keyword(['ZX300', self.date], save_dir=f'{self.raw_target_path}/{self.date}', file_extensions=['.csv'], limit=50)
                            manager.logout()
                            os.rename(f'{self.raw_target_path}/{self.date}/ZX300_weight_{self.date}.csv', f'{self.raw_target_path}/{self.date}/{self.date}_CZQZX300_000300.XSHG_000906.XSHG_target.csv')

                        except Exception as e:
                            log_message(f"--账户：{acct} {acct_chinese_name} 下载陈志强权重邮件出错: {e}", script_name=script_name, log_level=logging.warning)

                acct_chinese_name = acct_info.get('acct_chinese_name', '')
                log_message(f'--账户：{acct} {acct_chinese_name}', script_name=script_name)
                try:
                    self.process_split(acct, acct_info)
                except Exception as e:
                    log_message(f'--账户：{acct} {acct_chinese_name}，出错：{e}', script_name=script_name, log_level=logging.warning)


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))
    with open(f'{root_path}/split_system.yaml', 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)

    date = config.get('date')
    if date == 'current':
        date = dt.datetime.now().strftime('%Y%m%d')
    SS = SplitSystem(date, config)
    SS.main()