import pandas as pd
import pulp
import datetime as dt
from  stock_data_client import StockDataClient


class IndustryOptimizer:
    def __init__(self, stock_data: pd.DataFrame, benchmark_industry_weights: pd.DataFrame, direction: int, style_exposures:pd.DataFrame=None,
                 benchmark_style_exposures:pd.DataFrame=None):
        """
        初始化带权重分配的IndustryOptimizer
        :param stock_data: DataFrame，包含列 'order_book_id', 'industry', 'score'
        :param benchmark_industry_weights: float，指数成分股行业权重
        :param direction: 优化方向，最大化、最小化
        """
        self.stock_data = stock_data.copy()
        self.benchmark_industry_weights = benchmark_industry_weights.copy()
        self.direction = direction

    def optimize(self, max_stock_weight=0.02, turnover_limit=0.2, min_industry_limit=0.2, max_industry_limit=0.5, prev_weights: dict = None):
        """
        使用线性规划求解优化问题，返回权重分配
        """
        stock_data = self.stock_data.reset_index()
        if self.direction == 1:
            prob = pulp.LpProblem("IndustryOptimizer", pulp.LpMaximize)
        elif self.direction == -1:
            prob = pulp.LpProblem("IndustryOptimizer", pulp.LpMinimize)
        else:
            raise ValueError(
                f"direction:{self.direction}是非法值，请确认输入IndustryOptimizer(stock_data, benchmark_industry_weights, direction)")
        stocks = stock_data['order_book_id'].tolist()
        scores = dict(zip(stock_data['order_book_id'], stock_data['score']))

        ####
        stock_weights = self.benchmark_industry_weights['weight'].to_dict()
        # 创建连续型权重变量，使用指数成分权重和max_stock_weight中的较大值作为上限
        weights = {}
        for stock in stocks:
            # 获取该股票的指数成分权重，如果没有则使用max_stock_weight
            stock_weight = stock_weights.get(stock, max_stock_weight)
            # 使用指数成分权重和max_stock_weight中的较大值作为上限
            upper_bound = max(stock_weight, max_stock_weight)
            if upper_bound > max_stock_weight:
                upper_bound = upper_bound * 1.5
            weights[stock] = pulp.LpVariable(
                f"v_{stock}", lowBound=0, upBound=upper_bound, cat='Continuous')
        # # 创建连续型权重变量（允许部分持仓）
        # weights = pulp.LpVariable.dicts("v", stocks, lowBound=0, upBound=max_stock_weight, cat='Continuous')

        ####

        # 创建连续型权重变量（允许部分持仓）
        # weights = pulp.LpVariable.dicts(
        #     "v", stocks, lowBound=0, upBound=max_stock_weight, cat='Continuous')

        # 目标函数：最大化加权评分
        prob += pulp.lpSum(weights[stock] * scores[stock] for stock in stocks)

        # 约束：总权重为100%，如果停牌、涨停、跌停股票过多，则按照实际权重和进行约束
        industry_weights_sum = self.benchmark_industry_weights.groupby('industry')[
            'weight'].sum().sum()
        prob += pulp.lpSum(weights[stock] for stock in stocks) == 1.0

        # 行业权重约束
        industry_info = dict()
        industry_weights = self.benchmark_industry_weights.groupby('industry')[
            'weight'].sum().to_dict()
        # print(f'industry_weights: {industry_weights}')
        for industry, industry_weight in industry_weights.items():
            min_weight = industry_weight - industry_weight * min_industry_limit
            if min_weight < 0.01:
                min_weight = 0.0
            max_weight = industry_weight + industry_weight * max_industry_limit
            if max_weight < 0.01:
                max_weight = 0.01
            # 确保权重范围合理
            min_weight = round(min_weight, 4)
            max_weight = round(min(max_weight, 1.0), 4)
            # 直接通过索引赋值（需确保索引层级正确）
            industry_info[industry] = {
                'min_weight': min_weight, 'max_weight': max_weight}
        # print(f'industry_info: {industry_info}')
        for industry, info in industry_info.items():
            industry_stocks = stock_data[stock_data['industry']
                                         == industry]['order_book_id']
            prob += pulp.lpSum(weights[stock]
                               for stock in industry_stocks) >= info['min_weight']
            prob += pulp.lpSum(weights[stock]
                               for stock in industry_stocks) <= info['max_weight']

        # 换手率约束
        if prev_weights is not None:
            z = {stock: pulp.LpVariable(f"z_{stock}", lowBound=0)
                 for stock in stocks}
            for stock in stocks:
                old_w = prev_weights.get(stock, 0.0)
                prob += weights[stock] - old_w <= z[stock]
                prob += old_w - weights[stock] <= z[stock]
            # 总绝对变动 ≤ 2 × turnover_limit
            prob += pulp.lpSum(z.values()) <= 2 * turnover_limit

        # 求解问题
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != 'Optimal':
            raise ValueError(
                f"{dt.datetime.now().strftime('%Y%m%d')} 未找到可行解，{pulp.LpStatus[prob.status]} 请放宽约束条件 max_stock_weight:{max_stock_weight} turnover_limit:{turnover_limit} industry_limit:{min_industry_limit} {max_industry_limit} 行业权重和：{self.benchmark_industry_weights.groupby('industry')['weight'].sum().sum()}")
        # 提取结果，剔除权重小于0.01的股票
        # print({stock: weights[stock].value(
        # ) for stock in stocks if weights[stock].value() <= 0.00001})
        results = {stock: weights[stock].value(
        ) for stock in stocks if weights[stock].value() > 0.00001}
        # 构造返回结果
        selected_df = stock_data[stock_data['order_book_id'].isin(
            results.keys())].copy()
        selected_df['weight'] = selected_df['order_book_id'].map(results)
        selected_df = selected_df.sort_values(by='industry', ascending=False)
        result_df = selected_df.set_index(['order_book_id'])

        return result_df


# 示例用法
if __name__ == "__main__":

    end_date = '2025-06-18'

    data_source_path = r"/home/samba/Market"

    bench_mark = '000300.XSHG'
    client = StockDataClient(data_path=data_source_path)
    benchmark_industry_weights = client.get_stock_index_comments_weights_industry(
        bench_mark, start=end_date, end=end_date)
    benchmark_industry_weights = benchmark_industry_weights.loc[end_date]
    print(benchmark_industry_weights)

    order_book_ids = benchmark_industry_weights.index.get_level_values(
        'order_book_id').unique().tolist()

    factor_file_path = f"/home/trading/pred_df/df_test_PB_ScorpioV4_2020_20250618.parquet"
    df_factor = pd.read_parquet(factor_file_path).set_index(
        ['date', 'order_book_id']).sort_index()
    df_factor = df_factor.loc[end_date]
    factor_name = 'Scorpio_ezavg_grouped_scaled'
    df_factor['score'] = df_factor[factor_name]
    print(df_factor)
    df_factor = pd.concat(
        [df_factor, benchmark_industry_weights], axis=1, join="inner")
    df_factor.name = factor_name
    print(df_factor)

    # 过滤停牌、涨停、跌停股票
    suspended_df = client.get_stock_suspended_info(
        order_book_ids, start=end_date, end=end_date)
    suspended_df = suspended_df.stack().reset_index()
    suspended_df.rename(
        columns={'level_1': 'order_book_id', 0: 'suspended'}, inplace=True)
    suspended_df = suspended_df[suspended_df['suspended'] == False]
    suspended_df = suspended_df.set_index(['order_book_id'])

    order_book_ids = suspended_df.index.get_level_values(
        'order_book_id').unique().tolist()
    data_df = client.get_stock_post_1d_data(
        order_book_ids, start=end_date, end=end_date).swaplevel().sort_index().dropna()

    data_df['to_limit_up'] = data_df.eval(f'close >= {1 - 0.02} * limit_up')
    data_df['to_limit_down'] = data_df.eval(
        f'close <= {1 + 0.02} * limit_down')
    data_df = data_df[data_df['to_limit_up'] == False]
    data_df = data_df[data_df['to_limit_down'] == False]
    data_df = data_df.reset_index()
    print(data_df)

    filter_df = pd.DataFrame()
    filter_df['order_book_id'] = data_df['order_book_id']
    filter_df['flag'] = True
    filter_df = filter_df.set_index(['order_book_id'])

    # 行索引交集
    common_rows = df_factor.index.intersection(filter_df.index)
    # 提取共有部分
    df_factor = df_factor.loc[common_rows, :]
    print(df_factor)
    # 创建优化器实例
    optimizer = IndustryOptimizer(
        df_factor, benchmark_industry_weights=benchmark_industry_weights, direction=1)
    # 运行优化（限制单只股票最大权重为5%）
    selected_df = optimizer.optimize(max_stock_weight=0.0125, turnover_limit=0.15,
                                     min_industry_limit=0.5, max_industry_limit=2.0, prev_weights=None)
    print(selected_df)
    selected_data = selected_df.reset_index()
    print("order_book_id    score   industry    weight")
    for row in selected_data.itertuples():
        print(f"{getattr(row, 'order_book_id')} {getattr(row, 'score')} {getattr(row, 'industry')} {getattr(row, 'weight')}")
    # 计算行业分布
    industry_dist = df_factor.groupby(
        'industry')['weight'].sum().sort_values(ascending=False)
    print(f"{bench_mark}指数行业权重分布:{industry_dist.sum()}")
    print(industry_dist.round(4))
    industry_dist = selected_data.groupby(
        'industry')['weight'].sum().sort_values(ascending=False)
    print(f"选股结果行业权重分布:{industry_dist.sum()}")
    print(industry_dist.round(4))
    print("\n")
