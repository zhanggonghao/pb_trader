import os
import sys
import yaml
import rqdatac
import platform
import pandas as pd
import numpy as np
import datetime as dt
from ultis.dataproxy.rqfactors import add_factor
from ultis.email_manager import *
from ultis.stock_data_client import StockDataClient
# os.system(r'net use Z: \\192.168.1.168\samba /persistent:yes')
# client = StockDataClient(data_path=r'Z:\Market')
# client = StockDataClient(data_path=r'\\192.168.1.168\samba\Market')
client = StockDataClient(data_path=r'\\192.168.3.100\samba\Market')
import ultis.portfolio_optimizer_re_test as portfolio_optimizer
import ultis.factor_explosure_test as factor_explosure
import warnings
warnings.filterwarnings('ignore')
# rqdatac.init(13601611030,'PB123456789')
# rqdatac.init(username='18101949790', password='123456')
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)

#
class TransformTarget(object):
    def __init__(self, config, date):
        self.model_path = config.get('model_path')
        self.static_data_path = config.get('static_data_path')
        self.target_path = config.get('target_path')
        self.parameters = config.get('parameters')
        self.database_path = config.get('DATABASE_PATH') # 本地数据库路径
        self.blacklist_path = config.get('blacklist_path') # 黑名单路径
        self.date = date
        self._date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
        self.pre_date = rqdatac.get_previous_trading_date(
            date).strftime('%Y%m%d')  # 20250711
        print(self.pre_date)
        # 2025-07-11
        self._pre_date = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'
        # 下载黑名单邮件
        self.download_blackfile_from_email()
        # 获取因子暴露数据
        self.factor_explosure_param = config.get('factor_explosure_param')
        self.stock_factors, self.index_exposures = self.get_factor_explosure()

    # 下载黑名单邮件
    def download_blackfile_from_email(self):
        # 下载招商DMA黑名单
        manager = EmailManager()
        if not os.path.exists(f'{self.blacklist_path}/cms_long_short_black_list_{self.date}.xlsx'):
            # manager.download_attachments_by_keyword(["cms_long_short_black_list", self.date], save_dir='/home/trading/招商DMA交易表', file_extensions=['.xlsx'], limit=10)
            manager.download_attachments_by_keyword(["cms_long_short_black_list", self.date], save_dir=self.blacklist_path, file_extensions=['.xlsx'], limit=20)

        if not os.path.exists(f'{self.blacklist_path}/Restriction_List_{self.date}.xlsx'):
            manager.download_attachments_by_keyword(['Restriction List', self.date], save_dir=self.blacklist_path, file_extensions=['.xlsx'], limit=20)
            if not os.path.exists(f'{self.blacklist_path}/Restriction_List_{self.date}.xlsx'):
                manager.send_email_with_attachments(to=['15556235305@163.com', 'pagududeshengjiang@shpbjj.com'], subject="中信里昂黑名单缺失", body="未收到中信里昂黑名单邮件")
        manager.logout()

    # 获取因子暴露数据
    def get_factor_explosure(self):
        sample_range = self.factor_explosure_param.get('sample_range')
        bench_mark = self.factor_explosure_param.get('bench_mark')
        factor_start_date = self.factor_explosure_param.get('factor_start_date')
        factor_rolling_window = self.factor_explosure_param.get('factor_rolling_window')
        stock_factors, index_exposures = factor_explosure.main(sample_range, bench_mark, factor_start_date, self.database_path, factor_rolling_window)
        return stock_factors, index_exposures

    # 计算个股基准权重 + 动态单票上限（与 generate_index_weights.py 算法一致）
    def get_index_weight_limits(
            self,
            sample_range,
            bench_mark,
            stock_weight_multiplier,
            stock_weight_fallback,
            stock_weight_cap=None,
    ):
        """
        在本地按 generate_index_weights.py 的算法计算 sample_range 中每只股票相对
        bench_mark 总市值的市值加权权重，并构造动态单票上限：
            weight_i = exp(lcap_i) / sum(exp(lcap_k) in bench_mark on date)
            max_weight = np.maximum(weight * stock_weight_multiplier, stock_weight_fallback)
            若 stock_weight_cap 非 None，则再限制 max_weight <= stock_weight_cap

        实盘单日：start = end = pre_date。
        """
        try:
            return factor_explosure.compute_index_weight_limits(
                sample_range=sample_range,
                bench_mark=bench_mark,
                start=self._pre_date,
                end=self._pre_date,
                data_path=self.database_path,
                stock_weight_multiplier=float(stock_weight_multiplier),
                stock_weight_fallback=float(stock_weight_fallback),
                stock_weight_cap=None if stock_weight_cap is None else float(stock_weight_cap),
            )
        except Exception as exc:
            print(
                f'[tranform_target] 个股基准权重计算失败({sample_range} vs {bench_mark} @ {self._pre_date}): '
                f'{exc!r}，单票上限回退为固定权重上限'
            )
            return None

    # 获取benchmark行业和权重
    def get_benchmark_industry_weights_1(self, bench_mark):
        benchmark_weights = rqdatac.index_weights_ex(
            bench_mark, start_date=self._pre_date, end_date=self._pre_date, market='cn').loc[self._pre_date].reset_index()
        benchmark_codes = benchmark_weights['order_book_id'].unique().tolist()

        benchmark_industrys = rqdatac.get_instrument_industry(order_book_ids=benchmark_codes, date=self._pre_date)[
            'first_industry_name'].reset_index().rename(columns={'first_industry_name': 'industry'})

        benchmark_industry_weights = pd.merge(
            benchmark_weights, benchmark_industrys, on='order_book_id', how='outer')

        benchmark_industry_weights['date'] = self._pre_date
        benchmark_industry_weights = benchmark_industry_weights[[
            'date', 'order_book_id', 'weight', 'industry']]

        return benchmark_industry_weights
    # 获取benchmark行业和权重
    def get_benchmark_industry_weights(self, bench_mark):
        """
        从本地数据库读取 benchmark 的成分股权重和行业，口径与 backtest 一致：
            - 权重：stocks/basics/index_components_weights_{bench_mark}.feather
            - 行业：stocks/basics/all_instruments_level1_industry.feather

        返回长表：
            ['date', 'order_book_id', 'weight', 'industry']
        其中 date 固定为 self._pre_date（即目标交易日的前一交易日）。
        """
        client = StockDataClient(data_path=self.database_path)

        # 1) benchmark 当日官方权重（本地宽表 → 长表）
        benchmark_weights_wide = client.get_stock_index_comments_weights(
            bench_mark, start=self._pre_date, end=self._pre_date
        )
        benchmark_weights_wide = benchmark_weights_wide.copy()
        benchmark_weights_wide.index = pd.to_datetime(benchmark_weights_wide.index)
        benchmark_weights = (
            benchmark_weights_wide.reset_index()
            .rename(columns={'index': 'date'})
            .melt(id_vars='date', var_name='order_book_id', value_name='weight')
        )
        benchmark_weights['date'] = pd.to_datetime(benchmark_weights['date'])
        benchmark_weights['weight'] = pd.to_numeric(benchmark_weights['weight'], errors='coerce')
        benchmark_weights = benchmark_weights.dropna(subset=['date', 'order_book_id', 'weight'])
        benchmark_weights = benchmark_weights[benchmark_weights['weight'] > 0].reset_index(drop=True)

        benchmark_codes = benchmark_weights['order_book_id'].unique().tolist()

        # # 2) benchmark 成分股行业（本地日频长表）
        industry_path = os.path.join(
            self.database_path,
            'stocks/basics/all_instruments_level1_industry.feather',
        )
        benchmark_industrys = pd.read_feather(industry_path)
        benchmark_industrys['date'] = pd.to_datetime(benchmark_industrys['date'])
        benchmark_industrys = benchmark_industrys[
            (benchmark_industrys['date'] == pd.to_datetime(self._pre_date))
            & (benchmark_industrys['order_book_id'].isin(benchmark_codes))
        ][['order_book_id', 'industry']].drop_duplicates(subset=['order_book_id'], keep='last')

        benchmark_industry_weights = pd.merge(
            benchmark_weights, benchmark_industrys, on='order_book_id', how='outer')

        benchmark_industry_weights['date'] = pd.to_datetime(self._pre_date)
        benchmark_industry_weights = benchmark_industry_weights[[
            'date', 'order_book_id', 'weight', 'industry']]

        return benchmark_industry_weights

    # 获取黑名单股票列表
    def get_blacklist(self, black_list, forbid_lst):

        blacklist = []
        if black_list:
            black_list = black_list.replace('20250714', f'{self.date}')
            print(f'黑名单: {black_list}')
            data = pd.read_excel(f'{self.blacklist_path}/{black_list}')
            data['pair_id'] = data['pair_id'].apply(
                lambda x: str(x).split('@')[0])
            blacklist = data['pair_id'].unique().tolist()
        print(f'券商禁止名单: {forbid_lst}')
        print(f'交易黑名单: {list(set(blacklist + forbid_lst))}')
        return list(set(blacklist + forbid_lst))

    # 获取当日即将涨停的股票代码
    def get_near_limit_codes(self, codes):
        # 根据上一个交易日计算下一个交易日的涨跌停价格
        price_data = rqdatac.get_price(codes, start_date=self._pre_date, end_date=self._pre_date, frequency='1d',
                                       fields=None, adjust_type='pre', skip_suspended=False, market='cn', expect_df=True,
                                       time_slice=None).reset_index()

        # 计算next_lmt_up和next_lmt_down
        def calculate_limits(row):
            prefix = row['order_book_id'][:2]
            if prefix in ['68', '30']:
                next_lmt_up = row['close'] * 1.1985
                next_lmt_down = row['close'] * 0.801
            else:
                next_lmt_up = row['close'] * 1.0985
                next_lmt_down = row['close'] * 0.901
            return pd.Series([next_lmt_up, next_lmt_down])

        price_data[['next_lmt_up', 'next_lmt_down']] = price_data.apply(calculate_limits, axis=1)
        # 根据最新分钟数据过滤掉涨停板股票
        close_price = rqdatac.current_minute(codes).reset_index()
        merge_df = pd.merge(price_data[['order_book_id', 'next_lmt_up']], close_price[[
                            'order_book_id', 'close']], on='order_book_id', how='left')
        near_limit_up_merge_df = merge_df[merge_df['close']>= merge_df['next_lmt_up']].reset_index(drop=True)

        near_limit_up_codes = near_limit_up_merge_df['order_book_id'].unique().tolist()
        return near_limit_up_codes

    # 获取模型文件并进行筛选
    def get_model_data_and_deal(self, black_list, forbid_lst, sample_range=None):
        blacklist = self.get_blacklist(black_list, forbid_lst)
        # print(rf'{self.model_path}\df_test_PB_V0422_{self.pre_date}.parquet')
        model_data = pd.read_parquet(rf'{self.model_path}\df_test_PB_V0422_{self.pre_date}.parquet')
        model_data = model_data.reset_index(drop=True).rename(columns={'Scorpio_ezavg_grouped_scaled': 'score'})
        # 过滤黑名单
        model_data = model_data[~model_data['order_book_id'].isin(blacklist)].reset_index(drop=True)
        # 获取成交量和市值
        sorted_data = add_factor(model_data, 'total_turnover_ma10', date_field='date', asset_field='order_book_id')
        sorted_data['prev_date'] = sorted_data['date'].apply(rqdatac.get_previous_trading_date)
        sorted_data['prev_date'] = pd.to_datetime(sorted_data['prev_date'])
        sorted_data = add_factor(sorted_data, 'market_cap_3_ma3', date_field='prev_date', asset_field='order_book_id')
        filter_sorted_df = sorted_data[(sorted_data['total_turnover_ma10'] >= 25000000) & (sorted_data['market_cap_3_ma3'] >= 2500000000)].reset_index(drop=True)
        filter_sorted_df = sorted_data.copy()
        #
        # 选股范围
        if sample_range: 
            sample_range_df = client.get_stock_index_comments(sample_range, start=self._pre_date, end=self._pre_date).T
            sample_range_df = sample_range_df[sample_range_df[self._pre_date] == 1]
            sample_range_codes = sample_range_df.index.tolist()
            filter_sorted_df = filter_sorted_df[filter_sorted_df['order_book_id'].isin(sample_range_codes)].reset_index(drop=True)

        filter_sorted_df = filter_sorted_df.sort_values(by='score', ascending=False)
        filter_sorted_codes = filter_sorted_df['order_book_id'].unique().tolist()
        # 连接行业属性
        filter_sorted_industrys = rqdatac.get_instrument_industry(order_book_ids=filter_sorted_codes, date=self._pre_date)['first_industry_name'].reset_index().rename(columns={'first_industry_name': 'industry'})
        filter_sorted_df = pd.merge(filter_sorted_df, filter_sorted_industrys[['order_book_id', 'industry']], on='order_book_id', how='left')

        # 剔除停牌票
        stop_info = rqdatac.is_suspended(filter_sorted_codes, start_date=self._date, end_date=self._date, market='cn').T
        stop_info.columns = ['stop']
        stop_info = stop_info[stop_info['stop'] == True]
        stop_codes = stop_info.index.tolist()
        print(f'停牌票: {stop_codes}')
        filter_sorted_df = filter_sorted_df[~filter_sorted_df['order_book_id'].isin(stop_codes)].reset_index(drop=True)

        # 剔除ST的股票
        st_info = rqdatac.is_st_stock( filter_sorted_codes, self._date, self._date).T
        st_info.columns = ['is_st']
        st_info = st_info[st_info['is_st'] == True]
        st_codes = st_info.index.tolist()
        print(f'ST股票: {st_codes}')
        filter_sorted_df = filter_sorted_df[~filter_sorted_df['order_book_id'].isin(st_codes)].reset_index(drop=True)
        # 剔除低价股

        # # 剔除即将涨停票
        # near_limit_up_codes = self.get_near_limit_codes(filter_sorted_codes)
        # print(f'将要涨停的股票: {near_limit_up_codes}')
        # filter_sorted_df = filter_sorted_df[~filter_sorted_df['order_book_id'].isin(
        #     near_limit_up_codes)].reset_index(drop=True)

        return filter_sorted_df[['order_book_id', 'date', 'score', 'industry']]

    # 获取 上一交易日目标持仓的权重
    def get_pre_target_weights(self, bench_mark, sample_range, unique_id):
        pre_target_paths = f'{self.target_path}/{self.pre_date}/{self.pre_date}_{unique_id}_{bench_mark}_{sample_range}_target.csv'
        if os.path.exists(pre_target_paths):
            pre_target = pd.read_csv(pre_target_paths)
            pre_target['code'] = pre_target['code'].apply(lambda x: int(str(x)[:6]))
            pre_target['symbol'] = pre_target['code'].apply(lambda x: str(x).zfill(6))
            pre_target['exchange'] = pre_target['symbol'].apply(lambda x: "XSHG" if x[:1] == '6' else "XSHE")
            pre_target['symbol'] = pre_target['symbol'] + '.' + pre_target['exchange']
            pre_target = pre_target.set_index('symbol')
            pre_target_weights = pre_target['w'].to_dict()
        else:
            pre_target_weights = None
        
        return pre_target_weights

    # 线性规划
    def calculate_target(self, unique_id, sample_range, bench_mark, dealed_model_data, benchmark_industry_weights, max_stock_weight, turnover_limit, min_industry_limit, max_industry_limit, pre_target_weights, parameter, stock_factors, index_exposures, index_weight_limits=None, direction=1):
        # 初始化
        raw_target = portfolio_optimizer.main(
            parameter, dealed_model_data, pre_target_weights,
            style_df=stock_factors, bench_style_df=index_exposures,
            index_weight_limits=index_weight_limits,
        )

        raw_target = raw_target.sort_values(
            by='score', ascending=False).reset_index()

        target = raw_target[['order_book_id', 'weight']].rename(columns={'order_book_id': 'code', 'weight': 'w'}).sort_values(by='w', ascending=False).reset_index(drop=True)

        print(benchmark_industry_weights)
        selected_df = raw_target.copy()
        selected_df['sum'] = selected_df['score'] * selected_df['weight']
        print(
            f"选股结果：{len(selected_df)}  目标值:{selected_df.groupby('industry')['sum'].sum().sum()}")
        print("order_book_id    score   industry    weight")
        for row in selected_df.itertuples():
            print(f"{getattr(row, 'order_book_id')} {getattr(row, 'score')} {getattr(row, 'industry')} {getattr(row, 'weight')}")

        print(selected_df.groupby('industry')['weight'].sum())

        print(f'唯一识别号: {unique_id} \n\t target \n\t {target}')
        # 最新target目录
        target_path = f'{self.target_path}/{self.date}'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # target.to_csv(
        #     f'/{target_path}/{self.date}_{unique_id}_{bench_mark}_{sample_range}_target.csv', encoding='gbk')
        target.to_csv(os.path.join(target_path, f'{self.date}_{unique_id}_{bench_mark}_{sample_range}_target.csv'), encoding='gbk')

    def process_unique_id(self, unique_id, parameter):
        print(parameter)
        bench_mark = parameter.get('bench_mark')
        sample_range = parameter.get('sample_range')
        max_stock_weight = parameter.get('max_stock_weight')
        turnover_limit = parameter.get('turnover_limit')
        min_industry = parameter.get('min_industry')
        max_industry = parameter.get('max_industry')
        black_list = parameter.get('black_list')
        forbid_lst = parameter.get('forbid_lst')
        # 动态单票上限的两个系数（与 backtest 一致），用于本地计算 index_weight_limits
        stock_weight_multiplier = parameter.get('stock_weight_multiplier', 1.1)
        stock_weight_fallback = parameter.get(
            'stock_weight_fallback', max_stock_weight if max_stock_weight is not None else 0.02
        )

        # benchmark的行业和权重
        benchmark_industry_weights = self.get_benchmark_industry_weights(
            bench_mark)

        # 处理过的模型 文件，加上了分数和行业
        dealed_model_data = self.get_model_data_and_deal(
            black_list, forbid_lst, sample_range=sample_range)

        # 上一交易日目标持仓的权重，dic
        pre_target_weights = self.get_pre_target_weights(
            bench_mark, sample_range, unique_id)
        print(f'上一交易日目标持仓的权重: {len(pre_target_weights)}')
        print(f'上一交易日目标持仓的权重: {pre_target_weights}')

        # 个股基准权重 + 动态单票上限（本地按 generate_index_weights.py 算法计算）
        index_weight_limits = self.get_index_weight_limits(
            sample_range=sample_range,
            bench_mark=bench_mark,
            stock_weight_multiplier=stock_weight_multiplier,
            stock_weight_fallback=stock_weight_fallback,
        )

        # 计算目标
        self.calculate_target(
            unique_id, sample_range, bench_mark, dealed_model_data,
            benchmark_industry_weights, max_stock_weight, turnover_limit,
            min_industry, max_industry, pre_target_weights, parameter,
            self.stock_factors, self.index_exposures,
            index_weight_limits=index_weight_limits, direction=1,
        )

    def main(self):
        for uid in self.parameters:
            print(uid)
            param = self.parameters[uid]
            self.process_unique_id(uid, param)

        print(f'---end-------{dt.datetime.now()}')

    def main1(self, uid):
        print(uid)
        param = self.parameters[uid]
        self.process_unique_id(uid, param)

        print(f'---end-------{dt.datetime.now()}')


if __name__ == "__main__":
    print(f'---start-------{dt.datetime.now()}')
    root_path = os.path.dirname(os.path.abspath(__file__))
    with open(f'{root_path}/target_config_test.yaml', 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)
    # print(config)
    date = dt.datetime.now().strftime('%Y%m%d')  # 20250714
    # date = '20260430'
    print(date)
    TT = TransformTarget(config, date)
    TT.main()
    