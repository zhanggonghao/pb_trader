import pandas as pd
import numpy as np
import rqdatac
from datetime import datetime, time, timedelta
import os
from glob import glob
import dataproxy.rqfactors
import stock_data_client as stockdata
import industry_optimizer


# rqdatac.init(13601611030,'PB123456789', use_pool=True, max_pool_size=8)
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)



class TraderBase:
    def __init__(self, data_path:str, factor_file_path:str, factor_name:str, bench_mark:str, trader_data_path:str):
        self.data_path = data_path
        self.factor_file_path = factor_file_path
        self.factor_name = factor_name
        self.bench_mark = bench_mark
        self.trader_data_path =trader_data_path
        self.data_client = stockdata.StockDataClient(data_path=self.data_path)
        self.factor_data = None
        self.benchmark_industry_weights = None
        self.target_position_df = None
        self.target_stock_list = []
        self.prev_weights = None
        
        # 加载数据
        self.factor_data = pd.read_parquet(self.factor_file_path).set_index(['date', 'order_book_id']).sort_index()
        self.factor_data = self.factor_data.reset_index()
        self.factor_data['score'] = self.factor_data[self.factor_name]
        # 加载基准行业权重
        self.benchmark_industry_weights = None

    def _get_stock_index_comments_weights_industry(self, bench_mark:str, factor_day:str):
        df = rqdatac.index_weights_ex(bench_mark, start_date=factor_day, end_date=factor_day, market='cn')
        order_book_ids = df.loc[factor_day].index.get_level_values('order_book_id').unique().tolist()
        df_all_index_comp_weights = pd.DataFrame(index=['date', 'order_book_id'], columns=['date', 'order_book_id', 'weight', 'industry'])
        df_all_index_comp_weights = df_all_index_comp_weights.set_index(['date', 'order_book_id']).dropna()
        
        industries = rqdatac.get_instrument_industry(order_book_ids=order_book_ids, date=factor_day)['first_industry_name']
        data = pd.concat([df.loc[factor_day], industries], axis=1, join="inner")
        data['industry'] = data['first_industry_name']
        del data['first_industry_name']
        data['date'] = factor_day
        data = data.reset_index()
        data = data.set_index(["date", "order_book_id"])
        df_all_index_comp_weights = pd.concat([df_all_index_comp_weights, data])
        self.benchmark_industry_weights = df_all_index_comp_weights
        # 从历史数据获取
        # self.benchmark_industry_weights = self.data_client.get_stock_index_comments_weights_industry(bench_mark, start=factor_day, end=factor_day)

    def filter_data(self, factor_day:str, sample_range:str, filter:list=None):
        # 过滤数据
        if filter:
            self.factor_data = self.factor_data[~self.factor_data['order_book_id'].isin(filter)]
        # 约束成交量和市值
        sorted_data = dataproxy.rqfactors.add_factor(self.factor_data, 'total_turnover_ma10', date_field='date', asset_field='order_book_id')
        sorted_data['prev_date'] = sorted_data['date'].apply(rqdatac.get_previous_trading_date)
        sorted_data['prev_date'] = pd.to_datetime(sorted_data['prev_date'])
        sorted_data = dataproxy.rqfactors.add_factor(sorted_data, 'market_cap_3_ma3', date_field='prev_date', asset_field='order_book_id')

        filter_sorted_df = sorted_data[sorted_data['total_turnover_ma10']>=25000000].reset_index(drop=True)
        filter_sorted_df = filter_sorted_df[filter_sorted_df['market_cap_3_ma3']>=2500000000].reset_index(drop=True)
        self.factor_data = filter_sorted_df

        # 合并
        self.factor_data = self.factor_data.set_index(['date', 'order_book_id']).sort_index()
        self.factor_data = self.factor_data.loc[factor_day]
        self._get_stock_index_comments_weights_industry(self.bench_mark, factor_day)
        self.benchmark_industry_weights = self.benchmark_industry_weights.loc[factor_day]
        
        # 过滤选股范围
        sample_range_df = rqdatac.index_weights_ex(sample_range, start_date=factor_day, end_date=factor_day, market='cn')
        order_book_ids = sample_range_df.loc[factor_day].index.get_level_values('order_book_id').unique().tolist()
        self.factor_data = self.factor_data.reset_index()
        self.factor_data = self.factor_data[self.factor_data['order_book_id'].isin(order_book_ids)]
        self.factor_data.name = self.factor_name
        self.factor_data.set_index(['order_book_id'], inplace=True)
        self.factor_data = self.factor_data.sort_values(by='score', ascending=False)
        # 连接行业属性
        all_instruments_industry = self.data_client.get_all_instruments_industry_data().dropna()
        all_instruments_industry = all_instruments_industry[['industry']].loc[factor_day]
        self.factor_data = pd.concat([self.factor_data, all_instruments_industry], axis=1, join="inner")
        print(f"选股范围：{sample_range},按打分排序后:\n", self.factor_data)

        # 根据上一个交易日计算下一个交易日的涨跌停价格
        price_data = rqdatac.get_price(self.factor_data.index.tolist(), start_date = factor_day, end_date = factor_day, frequency='1d', 
                                    fields=None, adjust_type='pre', skip_suspended =False, market='cn', expect_df=True,
                                    time_slice=None)
        price_data = price_data.reset_index()
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
        close_price = rqdatac.current_minute(self.factor_data.index.tolist())

        close_price.index.name='code'
        price_all = close_price.reset_index()#行索引引进矩阵
        price_close = pd.DataFrame(price_all[['order_book_id','datetime','close']])
        price_close = pd.merge(price_close, price_data[['order_book_id', 'next_lmt_up']], on='order_book_id', how='left')
        self.factor_data = self.factor_data.merge(price_close[['order_book_id', 'close', 'next_lmt_up']], on=['order_book_id'], how='left')
        self.factor_data = self.factor_data[self.factor_data['close'] < self.factor_data['next_lmt_up']]

        print(f"停牌或涨停股票数量:{len(price_data) - len(self.factor_data)}")

    def load_prev_target_position(self, trddt_str:str):
        # 加载上一个交易日目标持仓
        try:
            prev_target_position_df = pd.read_csv(f"{self.trader_data_path}/{trddt_str}_{self.bench_mark}_目标持仓组合.csv", dtype={'证券代码': str, '市场': int})
            prev_target_position_df['exchange'] = prev_target_position_df["市场"].apply(lambda x: "XSHG" if x == 1 else "XSHE")
            prev_target_position_df['symbol'] = prev_target_position_df['证券代码'] + "." + prev_target_position_df['exchange']
            prev_target_position_df = prev_target_position_df.set_index('symbol')
            self.prev_weights = prev_target_position_df['相对权重'].to_dict()
            print(f"上一交易日目标持仓:\n")
            print(prev_target_position_df)
        except:
            self.prev_weights = None
            print(f"上一交易日目标持仓不存在")


    def calculate_target_position(self, max_stock_weight:float, turnover_limit:float, min_industry_limit:float, max_industry_limit:float):
        # 创建优化器实例
        optimizer = industry_optimizer.IndustryOptimizer(self.factor_data, benchmark_industry_weights=self.benchmark_industry_weights, direction=1)
        # 运行优化（限制单只股票最大权重为5%）
        selected_df = optimizer.optimize(max_stock_weight=max_stock_weight, turnover_limit=turnover_limit, min_industry_limit=min_industry_limit, 
                                        max_industry_limit=max_industry_limit, prev_weights=self.prev_weights)
        selected_df = selected_df.sort_values(by='score', ascending=False)
        selected_df = selected_df.reset_index()
        # 定义一个函数来根据股票代码结尾生成新的后缀
        def get_code_nums(order_book_id):
            if order_book_id.endswith('.XSHE'):
                return str(order_book_id[:6])
            elif order_book_id.endswith('.XSHG'):
                return str(order_book_id[:6])
            else:
                return 'Unknown'

        # 使用 apply 函数来生成新的后缀列
        selected_df['code_nums'] = selected_df['order_book_id'].apply(get_code_nums)

        # 定义一个函数来根据股票代码结尾生成新的后缀
        def get_mark_nums(code):
            if code.endswith('.XSHE'):
                return str(2)
            elif code.endswith('.XSHG'):
                return str(1)
            else:
                return 'Unknown'

        # 使用 apply 函数来生成新的后缀列
        selected_df['mark_nums'] = selected_df['order_book_id'].apply(get_mark_nums)

        # 导出当日目标持仓
        self.target_position_df = pd.DataFrame()
        self.target_position_df['证券代码'] = selected_df['code_nums']
        self.target_position_df['市场'] = selected_df['mark_nums']
        self.target_position_df['相对权重'] = selected_df['weight']
        
        self.target_position_df['证券代码']= self.target_position_df['证券代码'].astype(str)
        self.target_position_df['市场']= self.target_position_df['市场'].astype(int)

        self.target_position_df.to_csv(os.path.join(self.trader_data_path, f"{datetime.now().strftime('%Y%m%d')}_{self.bench_mark}_目标持仓组合.csv"), encoding='utf-8-sig', index=False)
        print(self.benchmark_industry_weights)
        selected_df['sum'] = selected_df['score'] * selected_df['weight']
        print(f"选股结果：{len(selected_df)}  目标值:{selected_df.groupby('industry')['sum'].sum().sum()}")
        print("order_book_id    score   industry    weight")
        for row in selected_df.itertuples():
            print(f"{getattr(row, 'order_book_id')} {getattr(row, 'score')} {getattr(row, 'industry')} {getattr(row, 'weight')}")
        
        print(selected_df.groupby('industry')['weight'].sum())

class ATXTrader(TraderBase):

    def __init__(self, data_path:str, factor_file_path:str, factor_name:str, bench_mark:str, trader_data_path:str):
        super().__init__(data_path, factor_file_path, factor_name, bench_mark, trader_data_path)
        self.current_position_df = None
        self.atx_target_df = None
        self.atx_final_df = None


    def get_current_position(self, trddt_str:str):
        ATX_Position_File_List = sorted(glob(f"{self.trader_data_path}/持仓查询_{trddt_str}*.xlsx"))
        if len(ATX_Position_File_List) > 0:
            ATX_position_file_path = ATX_Position_File_List[-1]
            print(f"当前持仓文件路径：{ATX_position_file_path}")
            # 读取ATX当前持仓
            current_position_df = pd.read_excel(ATX_position_file_path, dtype={'证券代码': str})
            current_position_df['持仓数量'] = current_position_df['持仓数量'].astype(int)
            current_position_df = current_position_df[current_position_df['持仓数量'] > 0]
            current_position_df = current_position_df[['交易市场','证券代码','持仓数量']]
            current_position_df['证券代码'] = current_position_df.apply(lambda row: f"{row['证券代码']}.SH" if row['交易市场'] == '上交所' else f"{row['证券代码']}.SZ", axis=1)
            current_position_df.drop(columns=['交易市场'], inplace=True)
            # 修改列名
            current_position_df.rename(columns={'证券代码': 'symbol', '持仓数量': 'orderQty'}, inplace=True)

            print(f"当前持仓数量：{len(current_position_df)}\n", current_position_df)
        else:
            ATX_position_file_path = f"not found {self.trader_data_path}/持仓查询_{trddt_str}*.xlsx)"
            print(f"当前持仓文件路径：{ATX_position_file_path}")
            current_position_df = pd.DataFrame(columns=['symbol','orderQty'])

        self.current_position_df = current_position_df
        return self.current_position_df


    def generate_atx_target_order_list(self, short_label:str, volume:int, total_value:float, account:str):
        target_position_df = self.target_position_df
        target_position_df['exchange'] = target_position_df["市场"].apply(lambda x: "XSHG" if x == 1 else "XSHE")
        target_position_df['symbol'] = target_position_df['证券代码'] + "." + target_position_df['exchange']
        target_stock_list = self.target_position_df['symbol'].tolist()
        target_position_df = target_position_df.set_index(['symbol'])

        # 获取股票的最近一分钟的价格数据
        stock_data = rqdatac.current_minute(target_stock_list, skip_suspended=False)
        stock_data = stock_data.reset_index()
        # 计算总市值
        if volume > 0:
            # 获取合约的最近一分钟的价格数据
            if_data = rqdatac.current_minute(short_label, skip_suspended=False)
            if_close = if_data.iloc[-1]['close']
            total_value = if_close * 300 * volume * 1.03
        else:
            current_position_df = self.current_position_df.copy()
            if current_position_df.empty:
                total_value = total_value
            else:
                order_book_ids = current_position_df['symbol'].tolist()
                order_book_ids = [stock.replace('SH', 'XSHG').replace('SZ', 'XSHE') for stock in order_book_ids] 
                price_data = rqdatac.current_minute(order_book_ids=order_book_ids, skip_suspended=False).reset_index()
                def convert_code(code):
                    return code.replace('XSHG', 'SH').replace('XSHE', 'SZ')
                price_data['order_book_id'] = price_data['order_book_id'].apply(convert_code)
                price_data = price_data.set_index(['order_book_id'])
                current_position_df = current_position_df.set_index(['symbol'])
                total_value = (price_data['close'] * current_position_df['orderQty']).sum()

        # 准备数据框架
        result_data = []
        # 代码转换函数
        def convert_code(code):
            return code.replace('XSHG', 'SH').replace('XSHE', 'SZ')
            
        result_data = []
        # 计算每只股票的目标股数和准备表格
        for index, row in stock_data.iterrows():
            original_code = row['order_book_id']
            converted_code = convert_code(original_code)
            close_price = row['close']
            stock_value = target_position_df.loc[original_code,'相对权重'] * total_value
            order_qty = round(stock_value / close_price / 100) * 100
            if converted_code.startswith('68'):
                if order_qty < 200:
                    order_qty = 200
            else:
                if order_qty < 100:
                    order_qty = 100
            # 填充数据
            result_data.append([
                'TWAP',                 # Strategy
                account,                # clientName
                'kf_twap_plus',         # orderType
                converted_code,         # symbol
                order_qty,              # orderQty
                1,                      # Side
                '',                     # EffTime (will be filled later)
                '',                     # ExpTime (will be filled later)
                1,                      # LimAction
                1,                      # AftAction
                '',                     # AlgoParam
                ''                      # Market
            ])

        # 创建DataFrame
        df_ATX_target_portfolio = pd.DataFrame(result_data, columns=[
            'Strategy', 'clientName', 'orderType', 'symbol', 'orderQty', 'Side', 'EffTime', 'ExpTime', 'LimAction', 'AftAction', 'AlgoParam', 'Market'
        ])

        # 设置 EffTime 和 ExpTime
        current_time = datetime.now()
        eff_time = (current_time + timedelta(minutes=10)).strftime('%H:%M:%S')
        exp_time = (current_time + timedelta(minutes=25)).strftime('%H:%M:%S')

        df_ATX_target_portfolio['EffTime'] = eff_time
        df_ATX_target_portfolio['ExpTime'] = exp_time

        # 导出CSV
        ATX_target_file_name  =  f"ATX_Target_{account}_{datetime.now().strftime('%Y%m%d')}_{self.bench_mark}_target_order_list.csv"
        df_ATX_target_portfolio.to_csv(self.trader_data_path + "/" + ATX_target_file_name, encoding='utf-8-sig', index=False)
        print(f"目标持仓: {len(df_ATX_target_portfolio)}\n", df_ATX_target_portfolio)
        self.atx_target_df = df_ATX_target_portfolio
        return df_ATX_target_portfolio

    def gennerate_atx_twap_final_order_list(self, account:str, start_time:str, end_time:str):
        tmp_target_df  = self.atx_target_df[['symbol','orderQty']]
        # 使用外连接
        merged_df = pd.merge(tmp_target_df, self.current_position_df, on='symbol', how='outer', suffixes=('_target', '_current'))
        # 填充缺失值为0
        merged_df = merged_df.fillna(0)
        # 计算第二列的差值
        merged_df['orderQty'] = (merged_df['orderQty_target'] - merged_df['orderQty_current']).round(-2)
        # 添加差值正负标识列
        merged_df['Side'] = np.where(merged_df['orderQty'] > 0, 1, 2)
        #格式化数据‘
        # merged_df.drop(columns=['orderQty_target','orderQty_current'], inplace=True)
        merged_df['orderQty'] = merged_df['orderQty'].abs()
        merged_df['orderQty'] = np.floor(merged_df['orderQty']).astype(int)
        # 四舍五入到 100 或 200
        def round_order_qty(row):
            if row['Side'] == 1:
                if row['symbol'].startswith('68'):
                    return np.round(row['orderQty'] / 200) * 200
                else:
                    return np.round(row['orderQty'] / 100) * 100
            else:
                return row['orderQty']

        merged_df['orderQty'] = merged_df.apply(round_order_qty, axis=1).astype(int)

        #如果orderQty为0则去除
        merged_df = merged_df[merged_df['orderQty'] != 0.0]
        # 设置 EffTime 和 ExpTime
        # current_time = datetime.now()
        # eff_time = (current_time + timedelta(minutes=10)).strftime('%Y%m%dT%H%M%S000')
        # exp_time = (current_time + timedelta(minutes=25)).strftime('%Y%m%dT%H%M%S000')
        # eff_time = current_time.replace(hour=9, minute=35, second=0, microsecond=0).strftime('%Y%m%dT%H%M%S000')
        # exp_time = current_time.replace(hour=9, minute=45, second=0, microsecond=0).strftime('%Y%m%dT%H%M%S000')
        # 批量填充数据
        merged_df['EffTime'] = start_time
        merged_df['ExpTime'] = end_time
        merged_df['Strategy'] = 'TWAP'
        merged_df['clientName'] = account
        merged_df['orderType'] = 'kf_twap_plus'
        merged_df['LimAction'] = 1
        merged_df['AftAction'] = 1
        merged_df['AlgoParam'] = ''
        merged_df['Market'] = ''
        merged_df = merged_df.reset_index(drop=True)
        self.atx_final_df = merged_df
        # 指定的列顺序
        desired_order = ['Strategy', 'clientName', 'orderType', 'symbol', 'orderQty', 'Side', 
                        'EffTime', 'ExpTime', 'LimAction', 'AftAction', 'AlgoParam', 'Market']
        # 根据卡方算法要求重新排列列
        ATX_KF_TWAP_df = merged_df[desired_order]

        # 导出CSV
        ATX_target_file_name  =  f"ATX_Target_{account}_{datetime.now().strftime('%Y%m%d')}_{self.bench_mark}_final_order_list.csv"
        ATX_KF_TWAP_df.to_csv(self.trader_data_path + "/" + ATX_target_file_name, encoding='utf-8-sig', index=False)

        print(f"调仓表: {len(ATX_KF_TWAP_df)}\n", ATX_KF_TWAP_df)

        return ATX_KF_TWAP_df

    def generate_atx_t0_order_list(self, account:str):
        hold_stock_list = self.current_position_df['symbol'].tolist()
        # 找到继续持有的股票
        matching_symbols = self.atx_final_df[self.atx_final_df['symbol'].isin(hold_stock_list)]
        # 过滤清仓卖出的股票
        matching_symbols = matching_symbols[matching_symbols['orderQty_target'] > 0]
        # 显示匹配的结果
        matching_symbols = matching_symbols.reset_index(drop=True)
        # 定义新的 DataFrame 的各列内容
        strategy = ['T0'] * len(matching_symbols)
        client_name = [account] * len(matching_symbols)
        ord_type = ['kf_t0'] * len(matching_symbols)
        symbol = matching_symbols['symbol'].values

        # 计算 OrderQty，先乘以 0.85
        def calculate_order_qty(row):
            if row['Side'] == 1:
                return row['orderQty_current'] * 0.90
            else:
                return row['orderQty_target'] * 0.90
        # 根据买卖计算买入前的持仓或卖出后的持仓，可以T0交易
        order_qty = matching_symbols.apply(calculate_order_qty, axis=1).astype(int)
        # 四舍五入到 100 或 200
        def round_order_qty(row):
            if row['symbol'].startswith('68'):
                return np.round(row['OrderQty'] / 200) * 200
            else:
                return np.round(row['OrderQty'] / 100) * 100

        order_qty = order_qty.apply(lambda x: np.round(x / 100) * 100)
        order_qty = pd.DataFrame({'symbol': symbol, 'OrderQty': order_qty})
        order_qty['OrderQty'] = order_qty.apply(round_order_qty, axis=1).astype(int)

        # 其他固定列的内容
        buyside = [1] * len(matching_symbols)
        sellside = [2] * len(matching_symbols)

        current_time = datetime.now()
        # EffTime and ExpTime
        eff_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0).strftime('%Y%m%dT%H%M%S000')
        exp_time = current_time.replace(hour=14, minute=55, second=0, microsecond=0).strftime('%Y%m%dT%H%M%S000')
        eff_time_list = [eff_time] * len(matching_symbols)
        exp_time_list = [exp_time] * len(matching_symbols)

        # LimAction, AftAction, AlgoParam
        lim_action = ['0'] * len(matching_symbols)
        aft_action = ['是'] * len(matching_symbols)
        algo_param = [''] * len(matching_symbols)

        # 交易市场
        def determine_market(sym):
            if sym.endswith('SZ'):
                return '深交所'
            elif sym.endswith('SH'):
                return '上交所'

        trading_market = [determine_market(sym) for sym in symbol]

        # 创建 DataFrame
        ATX_T0_df = pd.DataFrame({
            'Strategy': strategy,
            'ClientName': client_name,
            'OrdType': ord_type,
            'Symbol': symbol,
            'OrderQty': order_qty['OrderQty'].astype(int),
            'buyside': buyside,
            'sellside': sellside,
            'EffTime': eff_time_list,
            'ExpTime': exp_time_list,
            'LimAction': lim_action,
            'AftAction': aft_action,
            'AlgoParam': algo_param,
            '交易市场': trading_market
        })
        # 导出CSV
        ATX_T0_file_name  =  f"ATX_T0_{account}_{datetime.now().strftime('%Y%m%d')}_{self.bench_mark}_final_order_list.csv"
        ATX_T0_df.to_csv(self.trader_data_path + "/" + ATX_T0_file_name, encoding='utf-8-sig', index=False)
        print('T0交易表:\n', ATX_T0_df)
        return ATX_T0_df

class DMATrader(TraderBase):

    def __init__(self, data_path:str, factor_file_path:str, factor_name:str, bench_mark:str, trader_data_path:str):
        super().__init__(data_path, factor_file_path, factor_name, bench_mark, trader_data_path)
        self.current_position_df = None
        self.dma_target_df = None

    def get_current_position(self, trddt_str:str):
        ATX_Position_File_List = sorted(glob(f"{self.trader_data_path}/多空收益互换存续合约{trddt_str}*.xlsx"))
        if len(ATX_Position_File_List) > 0:
            ATX_position_file_path = ATX_Position_File_List[-1]
            print(f"当前持仓文件路径：{ATX_position_file_path}")
            # 读取当前持仓
            current_position_df = pd.read_excel(ATX_position_file_path, dtype={'证券代码': str})
            current_position_df['持仓数量'] = current_position_df['持仓数量'].astype(int)
            current_position_df=current_position_df[['交易对代码','持仓数量']]
            # 修改列名
            current_position_df.rename(columns={'交易对代码': 'symbol', '持仓数量': 'orderQty'}, inplace=True)
            print(f"当前持仓数量：{len(current_position_df)}\n", current_position_df)
        else:
            ATX_position_file_path = f"not found {broker_data_path}/多空收益互换存续合约{trddt}*.xlsx)"
            print(f"当前持仓文件路径：{ATX_position_file_path}")
            current_position_df = pd.DataFrame(columns=['symbol','orderQty'])
        self.current_position_df = current_position_df
        return current_position_df


    def generate_dma_target_order_list(self, short_label:str, volume:int, total_value:float, account:str):
        target_position_df = self.target_position_df
        target_position_df['exchange'] = target_position_df["市场"].apply(lambda x: "XSHG" if x == 1 else "XSHE")
        target_position_df['symbol'] = target_position_df['证券代码'] + "." + target_position_df['exchange']
        target_position_df = target_position_df.reset_index().set_index(['证券代码'])

        target_stock_list = target_position_df['symbol'].tolist()
        print(target_position_df)
        # 获取合约的最近一分钟的价格数据
        if_data = rqdatac.current_minute(short_label, skip_suspended=False)
        if_close = if_data.iloc[-1]['close']

        # 获取股票的最近一分钟的价格数据
        stock_data = rqdatac.current_minute(target_stock_list, skip_suspended=False)
        stock_data = stock_data.reset_index()

        # 计算总市值
        total_value = if_close * 300 * volume * 1.03
        # 准备数据框架
        result_data = []
        # 代码转换函数
        def convert_code(code):
            return code.replace('XSHG', 'XSHG@CNY').replace('XSHE', 'XSHE@CNY')
            
        result_data = []
        # 计算每只股票的目标股数和准备表格
        for index, row in stock_data.iterrows():
            original_code = row['order_book_id']
            converted_code = convert_code(original_code)
            close_price = row['close']
            stock_value = target_position_df.loc[original_code[0:6], '相对权重'] * total_value
            order_qty = round(stock_value / close_price / 100) * 100
            if converted_code.startswith('68'):
                if order_qty < 200:
                    order_qty = 200
            else:
                if order_qty < 100:
                    order_qty = 100
            # 填充数据
            result_data.append([
                account,                 # 资金账号
                converted_code,             # 标的代码
                order_qty,         # 申请数量
                '买入开仓',         # 买卖方向
                '',              # 起始时间
                '',                      # 截止时间
                'HX_SMART_TWAP',                     # 算法类型 
                '',                     # 算法参数 
                ''                      # 节点（多节点时必填）

            ])

        df_target_portfolio = pd.DataFrame(result_data, columns=[
            '资金账号', '标的代码', '申请数量', '买卖方向', '起始时间', '截止时间', '算法类型', '算法参数', '节点（多节点时必填）'])

        # 设置 EffTime 和 ExpTime
        current_time = datetime.now()
        eff_time = (current_time + timedelta(minutes=10)).strftime('%H:%M:%S')
        exp_time = (current_time + timedelta(minutes=25)).strftime('%H:%M:%S')

        df_target_portfolio['起始时间'] = eff_time
        df_target_portfolio['截止时间'] = exp_time
        self.dma_target_df = df_target_portfolio

        print(f"目标持仓: {len(df_target_portfolio)}\n", df_target_portfolio)
        return df_target_portfolio

    def gennerate_dma_twap_final_order_list(self, account:str, start_time:str, end_time:str):
        target_position_df = self.dma_target_df
        tmp_target_df  = target_position_df[['标的代码','申请数量']]
        tmp_target_df.rename(columns={'标的代码': 'symbol', '申请数量': 'orderQty'}, inplace=True)

        # 使用外连接
        merged_df = pd.merge(tmp_target_df, self.current_position_df, on='symbol', how='outer', suffixes=('_target', '_current'))
        # 填充缺失值为0
        merged_df = merged_df.fillna(0)
        # 计算第二列的差值
        merged_df['orderQty'] = ((merged_df['orderQty_target'] - merged_df['orderQty_current']) // 100) * 100
        # 添加差值正负标识列
        merged_df['Side'] = np.where(merged_df['orderQty'] > 0, 1, 2)
        
        #格式化数据‘
        merged_df.drop(columns=['orderQty_target','orderQty_current'], inplace=True)
        merged_df['orderQty'] = merged_df['orderQty'].abs()
        merged_df['orderQty'] = np.floor(merged_df['orderQty']).astype(int)
        # 去除股指
        merged_df = merged_df[merged_df['symbol'].apply(lambda x: not x.endswith("CCFX@CNY"))]

        # 四舍五入到 100 或 200
        def round_order_qty(row):
            if row['Side'] == 1:
                if row['symbol'].startswith('68'):
                    return np.round(row['orderQty'] / 200) * 200
                else:
                    return np.round(row['orderQty'] / 100) * 100
            else:
                return row['orderQty']

        merged_df['orderQty'] = merged_df.apply(round_order_qty, axis=1).astype(int)

        # 如果orderQty为0则去除
        merged_df = merged_df[merged_df['orderQty'] != 0.0]

        # 设置 EffTime 和 ExpTime
        # current_time = datetime.now()
        # eff_time = (current_time + timedelta(minutes=10)).strftime('%H:%M:%S')
        # exp_time = (current_time + timedelta(minutes=25)).strftime('%H:%M:%S')
        # eff_time = current_time.replace(hour=9, minute=40, second=0, microsecond=0).strftime('%H:%M:%S')
        # exp_time = current_time.replace(hour=9, minute=50, second=0, microsecond=0).strftime('%H:%M:%S')
        # 批量填充数据
        merged_df['资金账号'] = account
        merged_df['标的代码'] = merged_df['symbol']
        merged_df['申请数量'] = merged_df['orderQty']
        merged_df['买卖方向'] = merged_df['Side'].replace(1, '买入开仓').replace(2, '卖出平仓')
        merged_df['起始时间'] = start_time
        merged_df['截止时间'] = end_time
        merged_df['算法类型'] = 'HX_SMART_TWAP'
        merged_df['算法参数'] = ''
        merged_df['节点（多节点时必填）'] = ''

        desired_order = ['资金账号', '标的代码', '申请数量', '买卖方向', '起始时间', '截止时间', '算法类型', '算法参数', '节点（多节点时必填）']
        DMA_KF_TWAP_df = merged_df.reset_index(drop=True)[desired_order]
        # 导出CSV
        DMA_target_file_name  =  f"DMA_Target_{account}_{datetime.now().strftime('%Y%m%d')}_final_order_list.csv"
        DMA_KF_TWAP_df.to_csv(self.trader_data_path + "/" + DMA_target_file_name, encoding='utf-8-sig', index=False)

        print(f"调仓表: {len(DMA_KF_TWAP_df)}\n", DMA_KF_TWAP_df)
        return DMA_KF_TWAP_df

