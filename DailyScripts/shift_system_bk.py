import math
import os
import rqdatac
import pandas as pd
import numpy as np
import datetime as dt
from glob import glob
from ultis.format_client_algo import *
from ultis.log_msg import *
import warnings
warnings.filterwarnings('ignore')
# rqdatac.init()
# rqdatac.init(13601611030,'PB123456789', use_pool=True, max_pool_size=8)

rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)

# 获取当前脚本的文件名
script_name = os.path.basename(__file__)

# 匹配所需文件
def match_file(path, key, date):
    pos_files = sorted(glob(f"{path}/{date}_{key}*_target.csv"))
    if len(pos_files) == 0:
        return 
    else:
        return pos_files[-1]


class ShiftSystem(object):
    def __init__(self, date, config):
        self.date = date
        self.pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d') # 20250711
        self.raw_target_path = config.get('raw_target_path')
        self.standard_path = config.get('standard_path')
        self.out_path = config.get('out_path')
        self.product_info = config.get('product_info')
        self.stk_mkt_close = self.stk_mkt_close()  
        self.fut_close = self.get_fut_close()  
    
    #  获取股票昨收价格
    def stk_mkt_close(self):
        stk_mkt_codes = rqdatac.all_instruments(type='CS', market='cn', date=self.date)['order_book_id'].unique().tolist()
        stk_mkt_df = rqdatac.get_price(stk_mkt_codes, start_date=self.pre_date, end_date=self.pre_date, frequency='1d', fields=None, adjust_type='pre', skip_suspended =False, market='cn', expect_df=True, time_slice=None).reset_index()
        stk_mkt_df = stk_mkt_df[['order_book_id', 'close']].rename(columns={'order_book_id': 'code'})
        stk_mkt_df['code'] = stk_mkt_df['code'].apply(lambda x: int(x.split('.')[0]))
        return stk_mkt_df

    # 获取昨收持仓
    def get_prepos(self, raw_acct, target_acct):
        raw_pos_paths = f'{self.standard_path}/pos/{self.pre_date}/{self.pre_date}_{raw_acct}_pos.csv'
        if os.path.exists(raw_pos_paths):
            raw_pos = pd.read_csv(raw_pos_paths, index_col=0)
            raw_pos['label'] = 'raw'
        else:
            raw_pos = pd.DataFrame(columns=['code', 'hold', 'label'])


        target_pos_paths = f'{self.standard_path}/pos/{self.pre_date}/{self.pre_date}_{target_acct}_pos.csv'
        if os.path.exists(target_pos_paths):
            target_pos = pd.read_csv(target_pos_paths, index_col=0)
            target_pos['label'] = 'target'
        else:
            target_pos = pd.DataFrame(columns=['code', 'hold', 'label'])
        pos = pd.concat([raw_pos, target_pos]).reset_index(drop=True)
        return pos
        
    # 获取原始target
    def get_raw_target(self, uid):
        target_path = f'{self.raw_target_path}/{self.date}'
        target_paths = match_file(target_path, uid, self.date)
        target = pd.read_csv(target_paths, index_col=0)
        target['code'] = target['code'].apply(lambda x: int(str(x)[:6]))
        return target

    # 合并昨收持仓和和target
    def combine_pos_target(self, raw_acct, target_acct, uid, adjust_value, pre_reduce_value=0, pre_build_value=0):
        combine_pos = self.get_prepos(raw_acct, target_acct)
        prepos = combine_pos[['code', 'hold']].groupby('code').sum().reset_index()
        raw_target = self.get_raw_target(uid)
        merge_df = pd.merge(prepos, raw_target, on='code', how='outer')
        merge_df = pd.merge(merge_df, self.stk_mkt_close, on='code', how='left')
        prepos_value = (merge_df['hold'] * merge_df['close']).sum()
        log_message(f'--昨收持仓市值：{prepos_value}', script_name=script_name)
        log_message(f'--手动调整市值：{adjust_value}', script_name=script_name)
        merge_df['per_value'] = (prepos_value + adjust_value) * merge_df['w']
        merge_df['target'] = (merge_df['per_value'] / merge_df['close'])
        merge_df = merge_df.fillna(0)
        merge_df['target'] = merge_df['target'].round(0)
        merge_df['adj'] = merge_df['target'] - merge_df['hold']
        print(merge_df)
        exit()
        

        def func1(code, hold, close, vol):
            if code >= 688000:
                if vol >= 0:
                    tvol = max(200, vol) if vol >= 100 else (200 if close >= 100 else 0)
                else:
                    tvol = (
                        max(min(-200, vol), -hold)
                        if vol <= -100
                        else (-hold if vol <= -hold else 0)
                    )
            else:
                if vol > 0:
                    tvol = (
                        round(vol / 100, 0) * 100 # todo
                        if vol >= 50
                        else (100 if close >= 100 else 0)
                    )
                else:
                    vol = round(vol / 100, 0) * 100
                    tvol = (
                        max(min(-100, vol), -hold)
                        if vol <= -50 
                        else (-hold if vol <= -hold else 0)
                    )
            return tvol
        
        merge_df['adj1'] = merge_df.apply(lambda row: func1(row['code'], row['hold'], row['close'], row['adj']), axis=1)

        merge_df['target1'] = merge_df['hold'] + merge_df['adj1']
        
        target_value = (merge_df['target1'] * merge_df['close']).sum()
        log_message(f'--目标持仓市值：{target_value}', script_name=script_name)

        # 预留减仓，先设置为0
        merge_df['reduce_vol'] = 0
        merge_df['reduce_vol1'] = 0

        # 建仓单
        log_message(f'--预备建仓市值：{pre_build_value}', script_name=script_name)
        merge_df['per_build_value'] = pre_build_value * merge_df['w']
        merge_df['build_vol'] = (merge_df['per_build_value'] / merge_df['close'])
        merge_df = merge_df.fillna(0)
        merge_df['build_vol'] = merge_df['build_vol'].round(0)
        merge_df['build_vol1'] = merge_df.apply(lambda row: func1(row['code'], row['hold'], row['close'], row['build_vol']), axis=1)
        build_value = (merge_df['build_vol1'] * merge_df['close']).sum()
        log_message(f'--建仓单市值：{build_value}', script_name=script_name)

        merge_df['left_vol'] = np.where(merge_df['adj1'] >= 0, merge_df['hold'] + merge_df['reduce_vol'], merge_df['hold'] + merge_df['adj1'] + merge_df['reduce_vol'])

        return merge_df

    # 拆分各部分单子
    def split_cangdan(self, data):
        adj_df = data[['code', 'adj1', 'close']]
        
        adj_df = adj_df[adj_df['adj1'] != 0].reset_index(drop=True)
        reduce_df = data[['code', 'reduce_vol1', 'close']]
        reduce_df = reduce_df[reduce_df['reduce_vol1'] != 0].reset_index(drop=True)
        build_df = data[['code', 'build_vol1', 'close']]
        build_df = build_df[build_df['build_vol1'] != 0].reset_index(drop=True)
        t0_df = data[['code', 'left_vol', 'close']]
        t0_df = t0_df[t0_df['left_vol'] != 0].reset_index(drop=True)
        return adj_df, reduce_df, build_df, t0_df

    # 格式化股票代码
    def format_code(self, code):
        '''
        in: 1,000001,000001.XSHG
        out: 000001.SZ
        '''
        code = int(str(code)[:6])
        code = str(code).zfill(6)
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
        return code

    # 获取昨收期货行情，收盘价和结算价
    def get_fut_close(self):
        fut_mkt_df = rqdatac.all_instruments(type='Future', market='cn', date=self.date)
        fut_mkt_df = fut_mkt_df[fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))].reset_index(drop=True)
        fut_mkt_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(columns={'order_book_id': 'code'})
        fut_mkt_codes = fut_mkt_df['code'].tolist()
        fut_mkt = rqdatac.get_price(fut_mkt_codes, start_date=self.pre_date, end_date=self.pre_date, frequency='1d', fields=None, adjust_type='pre', skip_suspended =False, market='cn', expect_df=True, time_slice=None).reset_index()
        fut_mkt = fut_mkt[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
        fut_mkt_df = pd.merge(fut_mkt_df, fut_mkt, on='code', how='left')
        return fut_mkt_df

    def process_split(self):
        product_info = self.product_info
        product_type = product_info.get('product_type')
        benchmark = product_info.get('benchmark')

        acct_info = product_info.get('acct_info')
        raw_acct_info = acct_info.get('raw')
        raw_acct = raw_acct_info.get('acct')
        target_acct_info = acct_info.get('target')
        target_acct = target_acct_info.get('acct')

        uid = product_info.get('target_uid')
        adjust_value = product_info.get('adjust_value')

        raw_rzrq = raw_acct_info.get('rzrq')
        target_rzrq = target_acct_info.get('rzrq')
        data = self.combine_pos_target(raw_acct, target_acct, uid, adjust_value)
        # data.to_csv('/home/zhanggh/DailyScripts/test.csv')
        pre_hold_value = (data['hold'] * data['close']).sum()
        adj_df, reduce_df, build_df, t0_df = self.split_cangdan(data)

        # adj
        if acct_info.get('is_adj'):
            adj_df['adj_value'] = adj_df['adj1'] * adj_df['close']
            adj_buy = adj_df[adj_df['adj1'] >= 0]['adj_value'].sum()
            adj_sell = adj_df[adj_df['adj1'] < 0]['adj_value'].sum()
            log_message(f'--调仓买入金额：{adj_buy}', script_name=script_name)
            log_message(f'--调仓卖出金额：{adj_sell}', script_name=script_name)
            adj_ratio = (adj_buy - adj_sell) * 0.5 / pre_hold_value
            log_message(f'--调仓换手率：{format(adj_ratio, ".2%")}', script_name=script_name)
            format_adj_algo(self.out_path, adj_df, self.date, acct_info.get('adj_client'), acct_info.get('adj_acct'), acct_info.get('adj_algo'), acct, acct_info.get('adj_algo_starttime'), acct_info.get('adj_algo_endtime'), rzrq=is_rzrq)
        

        # reduce

        # build

        # t0
        if acct_info.get('is_t0'):
            t0_df['t0_value'] = t0_df['left_vol'] * t0_df['close']
            t0_amount = t0_df['t0_value'].sum()
            log_message(f'--可t0股票数量{len(t0_df)}', script_name=script_name)
            log_message(f'--可t0股票市值：{t0_amount}', script_name=script_name)
            format_t0_algo(self.out_path, t0_df, self.date, acct_info.get('t0_client'), acct_info.get('t0_acct'), acct_info.get('t0_algo'), acct, acct_info.get('t0_algo_starttime'), acct_info.get('t0_algo_endtime'))
        
        
    def main(self):
        for product in self.product_info:
            product_info = self.product_info.get(product)
            product_chinese_name = self.product_info.get('product_chinese_name')
            log_message(f'产品：{product} {product_chinese_name}', script_name=script_name)

            self.process_split()


if __name__ == "__main__":
    root_path = os.path.dirname(os.path.abspath(__file__))

    config = {
        'raw_target_path': '/home/zhanggh/TransformTargetData/target', # 权重target目录
        'standard_path': '/home/zhanggh/DailyScripts/TradeData/standarddata', # 交易相关标准数据目录
        'out_path': '/home/zhanggh/DailyScripts/SplitSystemData', # 输出目录
        'product_info':{
            'product': 'PBPFZX1H',
            'product_chinese_name': '配邦鹏飞中性1号',
            'product_type': 'zx',
            'benchmark': 'hs300',
            'target_uid': 'WBZD',
            'adjust_value': 0,
            'acct_info':{
                # 移出账户
                'raw':{
                    'acct': 'gtja_PBPFZX1H',
                    'acct_chinese_name': '国泰君安配邦鹏飞中性1号',
                    'acct_num': '10381131',
                    'is_adj': True,
                    'adj_client': 'ATX',
                    'adj_acct': '配邦鹏飞中性策略证券',
                    'adj_algo': 'KF',
                    'adj_algo_starttime': '093500',
                    'adj_algo_endtime': '095500',
                    'is_t0': True,
                    't0_client': '道和方舟',
                    't0_acct': '配邦鹏飞中性策略证券',
                    't0_algo': 'YR',
                    't0_algo_starttime': '093000',
                    't0_algo_endtime': '145500'},
                'target':{
                    'acct': 'haitong_PBPFZX1H',
                    'acct_chinese_name': '海通配邦鹏飞中性1号',
                    'acct_num': '5560000923',
                    'is_adj': True,
                    'adj_client': 'ATX',
                    'adj_acct': 'peibang-pt-71-pfzxcl',
                    'adj_algo': 'KF',
                    'adj_algo_starttime': '093500',
                    'adj_algo_endtime': '095500',
                    'is_t0': True,
                    't0_client': '道和方舟',
                    't0_acct': 'peibang-pt-71-pfzxcl',
                    't0_algo': 'YR',
                    't0_algo_starttime': '093000',
                    't0_algo_endtime': '145500'},
                }
    }}

    print(config)

    date = dt.datetime.now().strftime('%Y%m%d')
    SS = ShiftSystem(date, config)
    SS.main()
