import os
from tkinter import N
import rqdatac
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import yaml
import shutil
# rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=")
# rqdatac.init(13601611030, 'PB123456789', use_pool=True, max_pool_size=8)

'''
每日收益分析，beta，stk_lng_alpha, fut_lng_alpha, stk_sht_alpha, fut_sht_alpha, opt_alpha

'''


class DailyPnl(object):
    def __init__(self, date):
        self.date = date
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.product_acct_info = self.get_config()
        self.stk_price_info = self.get_stk_price_info()
        self.fut_price_info = self.get_fut_price_info()

    # 获取配置文件中
    def get_config(self):
        # 下载配置文件
        if not os.path.exists(f'{self.root_path}/PnlData/config/{self.date}_split_system.yaml'):
            shutil.copy(f'{self.root_path}/split_system.yaml',
                        f'{self.root_path}/PnlData/config/{self.date}_split_system.yaml')

        with open(f'{self.root_path}/PnlData/config/{self.date}_split_system.yaml', 'r', encoding='utf-8') as y:
            config = yaml.safe_load(y)

        return config.get('product_info')

    # 获取调仓时间区间
    def get_adj_periods(self):
        adj_periods = []
        for product in self.product_acct_info:
            for acct in self.product_acct_info.get(product).get('acct_info'):
                acct_info = self.product_acct_info.get(
                    product).get('acct_info').get(acct)
                adj_algo_starttime = acct_info.get('adj_algo_starttime')
                adj_algo_endtime = acct_info.get('adj_algo_endtime')
                adj_periods += [(adj_algo_starttime, adj_algo_endtime)]

        return list(set(adj_periods))

    # 获取股票当天全市场的vwap数据，包含昨收价，今收价，开盘半小时的vwap，全天vwap，以及调仓交易时段的vwap
    def get_stk_price_info(self):
        add_time_periods = self.get_adj_periods()
        time_periods = list(set([('093000', '100000')] + add_time_periods))
        print(time_periods)

        stk_mkt_codes = rqdatac.all_instruments(type='CS', market='cn', date=self.date)[
            'order_book_id'].unique().tolist()
        price_info = rqdatac.get_price(stk_mkt_codes, start_date=self.date, end_date=self.date, frequency='1d', fields=[
                                       'prev_close', 'close', 'total_turnover', 'volume'], adjust_type='pre', skip_suspended=False, market='cn', expect_df=True, time_slice=None).reset_index()
        price_info['allday_vwap'] = np.where(
            price_info['volume'] > 0, price_info['total_turnover'] / price_info['volume'], price_info['close'])
        price_info['date'] = price_info['date'].dt.strftime('%Y%m%d')
        price_info = price_info[['order_book_id',
                                 'date', 'prev_close', 'close', 'allday_vwap']]

        stk_mkt_df = rqdatac.get_price(stk_mkt_codes, start_date=self.date, end_date=self.date, frequency='1m', fields=None,
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
            price_info = pd.merge(price_info, grouped[['order_book_id', 'date', label]], on=[
                                  'order_book_id', 'date'])

        # print(price_info)
        price_info = price_info.rename(
            columns={'order_book_id': 'code', 'prev_close': 'preclose'})
        price_info['code'] = price_info['code'].apply(
            lambda x: int(str(x)[:6]))
        price_info['date'] = price_info['date'].apply(
            lambda x: str(x).replace('-', ''))
        return price_info

    # 获取股指期货/期权收盘价和结算价
    def get_fut_price_info(self):
        fut_mkt_df = rqdatac.all_instruments(
            type='Future', market='cn', date=self.date)
        fut_mkt_df = fut_mkt_df[(fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))) & (
            fut_mkt_df['symbol'].apply(lambda x: '连续' not in x))].reset_index(drop=True)
        fut_mkt_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(
            columns={'order_book_id': 'code'})
        fut_mkt_codes = fut_mkt_df['code'].tolist()
        fut_mkt = rqdatac.get_price(fut_mkt_codes, start_date=self.date, end_date=self.date, frequency='1d',
                                    fields=None, adjust_type='pre', skip_suspended=False, market='cn', expect_df=True, time_slice=None).reset_index()
        fut_mkt = fut_mkt[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(
            columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'})
        fut_mkt_df = pd.merge(fut_mkt_df, fut_mkt, on='code', how='left')
        return fut_mkt_df

    # 处理交易记录，股票多头收益，股票空头收益，股票多头vwap
    def deal_stk_order(self, accts, adj_fee_ratio=0.00015, t0_fee_ratio=0.0002):
        '''
        股票多头收益: （昨收价 - 成交价）* 成交数量 * 交易方向
        '''
        stk_lng_adj_slipvwap = 0
        stk_lng_adj_trade_act_pnl = 0
        stk_lng_t0_trade_act_pnl = 0
        stk_lng_t0_trade_expose_act_pnl = 0
        stk_sht_trade_pnl = 0

        for acct in accts:
            order_paths = f'{self.root_path}/PnlData/standarddata/entrust/{self.date}/{self.date}_{acct}_entrust.csv'

            if os.path.exists(order_paths):
                data = pd.read_csv(order_paths, index_col=0)
                data['date'] = data['date'].apply(lambda x: str(x))
                data['filled_amount'] = data['filled_vol'] * data['filled_price']
                data['fee'] = np.where(data['algo'] == 't0',
                                    np.where(data['dir'] == 1, data['filled_amount'] * t0_fee_ratio,
                                                data['filled_amount'] * (t0_fee_ratio + 0.0005)),
                                    np.where(data['dir'] == 1, data['filled_amount'] * adj_fee_ratio, data['filled_amount'] * (adj_fee_ratio + 0.0005)))
                data = pd.merge(data, self.stk_price_info, on=['code', 'date'], how='left')
                print(data)
                adj_data = data[data['algo'] == 'adj'].reset_index(drop=True)
                adj_data['adj_benchmark'] = adj_data.apply(
                    lambda row: row.get(f"{row['time_period']}_vwap"), axis=1)
                adj_data['benchmark_slipvwap'] = np.where(
                    adj_data['filled_price'] != 0,
                    (adj_data['filled_price'] /
                    adj_data['adj_benchmark'] - 1) * 10000 * adj_data['dir'] * -1,
                    0)
                adj_data['benchmark_slipvwap_amount'] = adj_data['benchmark_slipvwap'] * \
                    adj_data['filled_amount']
                # 调仓加权滑点
                _stk_lng_adj_slipvwap = adj_data['benchmark_slipvwap_amount'].sum(
                ) / adj_data['filled_amount'].sum() if adj_data['filled_amount'].sum() != 0 else 0

                # 调仓收益，策略收益
                adj_data['pnl'] = (adj_data['preclose'] - adj_data['filled_price']) * adj_data['filled_vol'] * adj_data['dir']
                adj_pnl = adj_data['pnl'].sum().round(0)
                adj_fee = adj_data['fee'].sum().round(0)
                _stk_lng_adj_trade_act_pnl = adj_pnl - adj_fee

                # T0收益
                t0_data = data[data['algo'] == 't0'].reset_index(drop=True)
                t0_data['cs_f'] = t0_data['filled_amount'] * t0_data['dir'] * -1
                tmp_t0_data = t0_data.copy()
                tmp_t0_data['vol'] = tmp_t0_data['dir'] * tmp_t0_data['filled_vol']
                grouped = tmp_t0_data[['code', 'vol']].groupby('code').sum().reset_index()
                t0_close_codes = grouped[grouped['vol'] == 0]['code'].tolist()
                t0_unclose_codes = grouped[grouped['vol'] != 0]['code'].tolist()
                t0_data = t0_data[t0_data.code.isin(t0_close_codes)].reset_index(drop=True)
                t0_pnl = t0_data['cs_f'].sum().round(0)
                t0_fee = t0_data['fee'].sum().round(0)
                _stk_lng_t0_trade_act_pnl = t0_pnl - t0_fee
                _stk_lng_t0_trade_expose_act_pnl = 0
                _stk_sht_trade_pnl = 0

                stk_lng_adj_slipvwap += _stk_lng_adj_slipvwap
                stk_lng_adj_trade_act_pnl += _stk_lng_adj_trade_act_pnl
                stk_lng_t0_trade_act_pnl += _stk_lng_t0_trade_act_pnl
                stk_lng_t0_trade_expose_act_pnl += stk_lng_t0_trade_expose_act_pnl
                stk_sht_trade_pnl += _stk_sht_trade_pnl
            
        return stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl, stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl

    # 处理股票持仓信息
    def deal_stk_pos(self, accts):
        stk_lng_hold_value = 0
        stk_lng_hold_act_pnl = 0
        stk_sht_hold_value = 0
        stk_sht_hold_act_pnl = 0
        for acct in accts:
            pos_paths = f'{self.root_path}/PnlData/standarddata/pos/{self.date}/{self.date}_{acct}_pos.csv'
            if os.path.exists(pos_paths):
                data = pd.read_csv(pos_paths, index_col=0)
                data['date'] = self.date
                data = pd.merge(data, self.stk_price_info, on=['code', 'date'], how='left')
                data['hold_pnl'] = data['hold'] * (data['close'] - data['preclose'])
                data['hold_value'] = data['close'] * data['hold']
                _stk_lng_hold_value = data['hold_value'].sum().round(0)
                _stk_lng_hold_act_pnl = data['hold_pnl'].sum().round(0)
                _stk_sht_hold_value = 0
                _stk_sht_hold_act_pnl = 0
                stk_lng_hold_value += _stk_lng_hold_value
                stk_lng_hold_act_pnl += _stk_lng_hold_act_pnl
                stk_sht_hold_value += _stk_sht_hold_value
                stk_sht_hold_act_pnl += _stk_sht_hold_act_pnl

        return stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl

    # 处理期货交易记录，量化交易收益，cta交易收益
    def deal_fut_order(self, product):
        order_paths = f'{self.root_path}/PnlData/standarddata/fut_data/{self.date}/{self.date}_{product}_fut_order.csv'
        quant_order_pnl = 0
        quant_order_fee = 0
        cta_order_pnl = 0
        cta_order_fee = 0
        if os.path.exists(order_paths):
            data = pd.read_csv(order_paths, index_col=0)
            if data.shape[0] != 0:
                data = pd.merge(data, self.fut_price_info, on='code', how='left')
                quant_data = data[data['user'] == 'quant'].reset_index(drop=True)
                if quant_data.shape[0] != 0:
                    quant_data['order_pnl'] = quant_data['Direction'] * quant_data['filled_vol'] *  quant_data['contract_multiplier'] * (quant_data['presettlement'] - quant_data['filled_price'])
                    quant_order_pnl = quant_data['order_pnl'].sum()
                    quant_order_fee = quant_data['UsedFee'].sum()

                cta_data = data[data['user'] == 'cta'].reset_index(drop=True)
                if cta_data.shape[0] != 0:
                    cta_data['order_pnl'] = cta_data['Direction'] * cta_data['filled_vol'] *  cta_data['contract_multiplier'] * (cta_data['presettlement'] - cta_data['filled_price'])
                    cta_order_pnl = cta_data['order_pnl'].sum()
                    cta_order_fee = cta_data['UsedFee'].sum()
        return round(quant_order_pnl, 2), round(quant_order_fee, 2), round(cta_order_pnl, 2), round(cta_order_fee, 2)

    # 处理期货持仓信息
    def deal_fut_pos(self, product):
        pos_paths = f'{self.root_path}/PnlData/standarddata/fut_data/{self.date}/{self.date}_{product}_fut_pos.csv'
        print(pos_paths)
        quant_hold_value = 0
        quant_hold_pnl = 0
        quant_diff_pnl = 0
        cta_hold_value = 0
        cta_hold_pnl = 0
        cta_diff_pnl = 0
        if os.path.exists(pos_paths):
            data = pd.read_csv(pos_paths, index_col=0)
            if data.shape[0] != 0:
                data = pd.merge(data, self.fut_price_info, on='code', how='left')
                quant_data = data[data['user'] == 'quant'].reset_index(drop=True)
                if quant_data.shape[0] != 0:
                    quant_data['hold_value'] = quant_data['Direction'] * quant_data['vol'] * quant_data['contract_multiplier'] * quant_data['settlement']
                    quant_hold_value = quant_data['hold_value'].sum().round(0)
                    quant_data['hold_pnl'] = (quant_data['settlement'] - quant_data['presettlement']) * quant_data['vol'] * quant_data['contract_multiplier'] * quant_data['Direction']
                    quant_hold_pnl = quant_data['hold_pnl'].sum().round(0)
                    quant_data['diff_pnl'] = (quant_data['settlement'] - quant_data['close']) * quant_data['Direction'] * quant_data['vol'] * quant_data['contract_multiplier']
                    quant_diff_pnl = quant_data['diff_pnl'].sum().round(0)
                cta_data = data[data['user'] == 'cta'].reset_index(drop=True)
                if cta_data.shape[0] != 0:
                    cta_data['hold_value'] = cta_data['Direction'] * cta_data['vol'] * cta_data['contract_multiplier'] * cta_data['settlement']
                    cta_hold_value = cta_data['hold_value'].sum().round(0)
                    cta_data['hold_pnl'] = (cta_data['settlement'] - cta_data['presettlement']) * cta_data['vol'] * cta_data['contract_multiplier'] * cta_data['Direction']
                    cta_hold_pnl = cta_data['hold_pnl'].sum().round(0)
                    cta_data['diff_pnl'] = (cta_data['settlement'] - cta_data['close']) * cta_data['Direction'] * cta_data['vol'] * cta_data['contract_multiplier']
                    cta_diff_pnl = cta_data['diff_pnl'].sum().round(0)
        return quant_hold_value, quant_hold_pnl, quant_diff_pnl, cta_hold_value, cta_hold_pnl, cta_diff_pnl

    # 获取股票账户资金信息
    def get_stk_assets(self, accts):
        res = pd.DataFrame()
        for acct in accts:
            assets_paths = f'{self.root_path}/PnlData/standarddata/assets/{self.date}/{self.date}_{acct}_assets.csv'
            if os.path.exists(assets_paths):
                data = pd.read_csv(assets_paths, index_col=0)
                res = pd.concat([res, data])
        res = res.groupby(['date', 'acct_id']).sum().reset_index()

        stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl, stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl = self.deal_stk_order(accts)
        stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl = self.deal_stk_pos(accts)
        return res, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl, stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl, stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl

    # 期货账户资金信息
    def get_fut_assets(self, product):
        fut_assets_paths = f'{self.root_path}/PnlData/standarddata/fut_data/{self.date}/{self.date}_{product}_fut_assets.csv'
        if os.path.exists(fut_assets_paths):
            data = pd.read_csv(fut_assets_paths, index_col=0)
        else:
            data = pd.DataFrame(columns=['date', 'DynamicRights', 'Margin', 'Available', 'Deposit', 'Withdraw'])
            data['date'] = [self.date]
        return data.fillna(0)
    
    # 期货期权资金信息
    def get_opt_assets(self, product):
        opt_assets_paths = f'{self.root_path}/PnlData/standarddata/opt_data/{self.date}/{self.date}_{product}_opt_assets.csv'
        
        if os.path.exists(opt_assets_paths):
            data = pd.read_csv(opt_assets_paths, index_col=0)
        else:
            data = pd.DataFrame(columns=['总资产','可用资金','已占用保证金及冻结资金','估算浮盈','风险度1','平仓盈亏'])
            data['总资产'] = [0]
        return data.fillna(0)
    

    # 获取产品信息
    def get_product_info(self, product, product_info):
        # print(f'--- {product_info}')
        product_chinese_name = product_info.get('product_chinese_name')
        product_type = product_info.get('product_type')
        benchmark = product_info.get('benchmark')
        fut_info = product_info.get('fut_info')
        option_info = product_info.get('option_info')
        acct_info = product_info.get('acct_info')
        # print(f'------ {acct_info}')
        # 股票账户资金信息

        stk_assets, stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl,stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl, stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl = self.get_stk_assets(acct_info.keys())
        print(stk_assets)
        print(stk_lng_adj_slipvwap, stk_lng_adj_trade_act_pnl,stk_lng_t0_trade_act_pnl, stk_lng_t0_trade_expose_act_pnl, stk_sht_trade_pnl)
        print(stk_lng_hold_value, stk_lng_hold_act_pnl, stk_sht_hold_value, stk_sht_hold_act_pnl)

        # 期货账户资金信息
        fut_assets = self.get_fut_assets(product)
        print(fut_assets)
        quant_order_pnl, quant_order_fee, cta_order_pnl, cta_order_fee = self.deal_fut_order(product)
        print(quant_order_pnl, quant_order_fee, cta_order_pnl, cta_order_fee)
        quant_hold_value, quant_hold_pnl, quant_diff_pnl, cta_hold_value, cta_hold_pnl, cta_diff_pnl = self.deal_fut_pos(product)
        print(quant_hold_value, quant_hold_pnl, quant_diff_pnl, cta_hold_value, cta_hold_pnl, cta_diff_pnl)

        # 期权账户资金信息
        opt_assets = self.get_opt_assets(product)
        print(opt_assets)

        # 整体分为如下部分，
        # 1.多头超额，空头基差，区分理论和实际
        # 2.cta端
        # 3.期权段
        # 4.产品收益率，量化端超额，量化端基差，cta收益率
        # 5.各部分收益之和与净资产之差比较



        pass


    def main(self):
        # print(self.product_acct_info)
        for product in self.product_acct_info:
            if product != 'PBHSZX1H':
                continue 
            print(product)
            product_info = self.product_acct_info.get(product)
            self.get_product_info(product, product_info)
            exit()




if __name__ == "__main__":
    # date = dt.datetime.now().strftime('%Y%m%d')
    date = '20260304'
    SP = DailyPnl(date)
    SP.main()
