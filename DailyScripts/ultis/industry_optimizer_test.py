import pandas as pd
import pulp
import datetime as dt
# from  stock_data_client import StockDataClient


class IndustryOptimizer:
    def __init__(self, stock_data: pd.DataFrame, benchmark_industry_weights: pd.DataFrame, direction: int, style_exposures:pd.DataFrame=None,
                 benchmark_style_exposures:pd.DataFrame=None):
        """
        初始化带权重分配的IndustryOptimizer
        必要数据结构:
        - stock_data: index=['date','order_book_id']，列含:
            * 'score': 用于目标函数（direction=1最大化，-1最小化）
            * 'industry': 股票所属行业
        - benchmark_industry_weights: index=['date','order_book_id']，列含:
            * 'industry': 基准成分股行业
            * 'weight':   基准个股权重（按日）；按日 groupby('industry')['weight'].sum() 得行业权重
        可选:
        - style_exposures: index=['date','order_book_id']，多列风格暴露（如 'size','value','quality','beta' ...）
        - benchmark_style_exposures: index=['date']（或 MultiIndex 包含 'date'），列为风格名；值为指数当日风格暴露
        """
        self.stock_data = stock_data.copy()
        start_date = self.stock_data.index.get_level_values('date').min()
        end_date = self.stock_data.index.get_level_values('date').max()
        # benchmark_industry_weights = client.get_stock_index_comments_weights_industry(order_book_id='000300.XSHG', start=start_date, end=end_date)
        self.benchmark_industry_weights = benchmark_industry_weights.copy()
        self.direction = direction
        self.selected_list = list()
        self.date_list = self.stock_data.index.get_level_values('date').unique().tolist()
        self.style_exposures = style_exposures  # 多列风格暴露
        self.benchmark_style_exposures = benchmark_style_exposures  # 指数风格暴露（按日）
        self.relax_records = list()

    def optimize(self, factor_name, max_stock_weight=0.02, turnover_limit=0.2, min_industry_limit=0.2, max_industry_limit=0.5,
                 style_sigma_bands=None):
        """
        :param style_sigma_bands: 
            - (lo, hi): 全部风格统一的±Nσ带
            - dict: {style_name: (lo, hi)} 各风格单独±Nσ带（推荐，style_name需与 *_exposure 列一致）
            σ为指数暴露按列计算的252日滚动标准差
        """
        prev_weights = None
        self.relax_records = list()

        # 预计算指数风格暴露的 252日滚动标准差
        sigma_df = None
        if (self.benchmark_style_exposures is not None) and (style_sigma_bands is not None):
            bench = self.benchmark_style_exposures.copy()
            print('指数数据：\n',bench)
            if 'date' in getattr(bench, 'columns', []):
                bench['date'] = pd.to_datetime(bench['date'])
                bench = bench.set_index('date').sort_index()
            elif isinstance(bench.index, pd.MultiIndex) and 'date' in bench.index.names:
                bench = bench.reset_index().set_index('date').sort_index()
                bench = bench.drop(columns=[c for c in bench.columns if c in ['order_book_id','industry','weight']], errors='ignore')
            else:
                bench.index = pd.to_datetime(bench.index)
                bench = bench.sort_index()
            # 仅使用以 _exposure 结尾的风格列，并转为数值
            bench = bench[bench.columns].apply(pd.to_numeric, errors='coerce')
            sigma_df = bench.rolling(window=252, min_periods=252).std()
            print('指数市值暴露滚动标准差：\n', sigma_df)

        # 放宽策略：仅当日生效，下一天回到原参数
        relax_multiplier = 2.0     # 每次放大倍数
        max_relax_tries = 2        # 放宽尝试次数（0 表示不放宽，1 表示放大一次，依此类推）

        self.selected_list = []

        for di, _date in enumerate(self.date_list):
            print('正在处理日期：', _date)
            # print('原始股票数据：\n', self.stock_data)
            stock_data = self.stock_data.loc[_date].copy().reset_index()
            stock_data = stock_data[pd.notna(stock_data[factor_name])]
            if stock_data.empty:
                continue
            stocks = stock_data['order_book_id'].tolist()
            scores = dict(zip(stock_data['order_book_id'], stock_data[factor_name]))
            # print('当日股票打分：', scores)

            solved = False
            for attempt in range(max_relax_tries + 1):
                # 重新建模（当日局部模型）
                prob = pulp.LpProblem("IndustryOptimizer", pulp.LpMaximize if self.direction == 1 else pulp.LpMinimize)
                # stocks = stock_data['order_book_id'].tolist()
                # print(stock_data)
                # 移除平移后导致的NaN
                # stock_data = stock_data[pd.notna(stock_data[factor_name])]
                # print('股票数据：', stock_data)
                # stocks = stock_data['order_book_id'].tolist()
                # scores = dict(zip(stock_data['order_book_id'], stock_data[factor_name]))
                # print('当日股票打分：', scores)

                stock_weights = self.benchmark_industry_weights.loc[_date]['weight'].to_dict()
                weights = {}
                for stock in stocks:
                # 获取该股票的指数成分权重，如果没有则使用max_stock_weight
                    stock_weight = stock_weights.get(stock, max_stock_weight)
                    # 使用指数成分权重和max_stock_weight中的较大值作为上限
                    upper_bound = max(stock_weight, max_stock_weight)
                    if upper_bound > max_stock_weight:
                        upper_bound = upper_bound * 1.0
                    weights[stock] = pulp.LpVariable(f"v_{stock}", lowBound=0, upBound=upper_bound, cat='Continuous')

                # weights = pulp.LpVariable.dicts("v", stocks, lowBound=0, upBound=max_stock_weight, cat='Continuous')

                # 目标
                prob += pulp.lpSum(weights[s] * scores[s] for s in stocks)

                # 全投资
                prob += pulp.lpSum(weights[s] for s in stocks) == 1.0

                # 行业约束（围绕基准权重±比例带）
                industry_info = dict()
                industry_weights = self.benchmark_industry_weights.loc[_date].groupby('industry')['weight'].sum().to_dict()
                # print('股票行业权重约束\n', industry_weights)
                for ind, bw in industry_weights.items():
                    lo = bw - bw * min_industry_limit
                    if lo < 0.01:
                        lo = 0.0
                    hi = bw + bw * max_industry_limit
                    if hi < 0.01:
                        hi = 0.01
                    lo = round(lo, 4)
                    hi = round(hi, 4)
                    industry_info[ind] = {'lo': lo, 'hi': hi}
                for ind, info in industry_info.items():
                    ind_stocks = stock_data[stock_data['industry'] == ind]['order_book_id']
                    prob += pulp.lpSum(weights[s] for s in ind_stocks) >= info['lo']
                    prob += pulp.lpSum(weights[s] for s in ind_stocks) <= info['hi']
                    # print(f'行业约束：{ind}，{info["lo"]}, {info["hi"]}')

                # 风格约束：相对指数 ± Nσ
                eff_lo_sigma, eff_hi_sigma = None, None  # 用于记录当次
                used_bands = {}  # 记录每个风格当次使用的(lo_sigma, hi_sigma)
                if (self.style_exposures is not None) and (sigma_df is not None) and (style_sigma_bands is not None):
                    if _date not in self.style_exposures.index.get_level_values('date'):
                        raise ValueError(f"{_date} 缺少风格暴露数据")
                    exp_df = self.style_exposures.loc[_date].reindex(stocks).fillna(0.0)
                    try:
                        bench_row = self.benchmark_style_exposures.loc[_date]
                        sigmas = sigma_df.loc[_date]
                    except Exception:
                        bench_row, sigmas = None, None

                    if isinstance(bench_row, (pd.Series, pd.DataFrame)) and isinstance(sigmas, (pd.Series, pd.DataFrame)):
                        styles = [c for c in exp_df.columns if (c in getattr(bench_row, 'index', [])) and (c in getattr(sigmas, 'index', []))]
                        scale = (relax_multiplier ** attempt)     # 仅当日临时放宽
                        for style_name in styles:
                            # 选择该风格的基础带宽
                            base_pair = None
                            if isinstance(style_sigma_bands, dict):
                                base_pair = style_sigma_bands.get(style_name, None)
                            elif isinstance(style_sigma_bands, (list, tuple)) and len(style_sigma_bands) == 2:
                                base_pair = style_sigma_bands
                            if base_pair is None:
                                continue
                            base_lo, base_hi = float(base_pair[0]), float(base_pair[1])
                            lo_sigma, hi_sigma = base_lo * scale, base_hi * scale
                            # 为了兼容旧记录字段，保留一份最后一次循环的值
                            eff_lo_sigma, eff_hi_sigma = lo_sigma, hi_sigma
                            used_bands[style_name] = (lo_sigma, hi_sigma)

                            bench_val = float(bench_row[style_name]) if not isinstance(bench_row, pd.DataFrame) else float(bench_row[style_name].iloc[0])
                            sigma_val = float(sigmas[style_name]) if not isinstance(sigmas, pd.DataFrame) else float(sigmas[style_name].iloc[0])
                            if pd.isna(sigma_val) or sigma_val <= 0:
                                continue
                            lo_abs = bench_val - lo_sigma * sigma_val
                            hi_abs = bench_val + hi_sigma * sigma_val
                            beta = dict(zip(exp_df.index, exp_df[style_name]))
                            comb = pulp.lpSum(weights[s] * beta[s] for s in stocks)
                            prob += comb >= lo_abs
                            prob += comb <= hi_abs
                            print(f'{style_name}风格因子约束：，{lo_abs}, {hi_abs}')

                # 换手率约束：仅非首日启用
                use_turnover = (di > 0) and (prev_weights is not None) and (len(prev_weights) > 0) \
                               and (turnover_limit is not None) and (turnover_limit > 0)
                if use_turnover:
                    z = {s: pulp.LpVariable(f"z_{s}", lowBound=0) for s in stocks}
                    for s in stocks:
                        old_w = prev_weights.get(s, 0.0)
                        prob += weights[s] - old_w <= z[s]
                        prob += old_w - weights[s] <= z[s]
                    prob += pulp.lpSum(z.values()) <= 2 * turnover_limit

                # 求解
                solver = pulp.PULP_CBC_CMD(msg=False)
                prob.solve(solver)
                status = pulp.LpStatus[prob.status]
                self.relax_records.append({
                    'date': pd.to_datetime(_date),
                    'attempt': attempt,
                    'scale': float(relax_multiplier ** attempt),
                    'lo_sigma_used': eff_lo_sigma,
                    'hi_sigma_used': eff_hi_sigma,
                    'style_bands_used': used_bands,  # 新增：每个风格当次带宽
                    'status': status,
                    'n_stocks': len(stocks)
                })

                if status == 'Optimal':
                    results = {s: weights[s].value() for s in stocks if weights[s].value() > 1e-5}
                # if pulp.LpStatus[prob.status] == 'Optimal':
                #     results = {s: weights[s].value() for s in stocks if weights[s].value() > 1e-5}
                    sel = stock_data[stock_data['order_book_id'].isin(results.keys())].copy()
                    sel['date'] = _date
                    sel['weight'] = sel['order_book_id'].map(results)
                    sel = sel.sort_values(by='industry', ascending=False)
                    self.selected_list.append(sel)
                    prev_weights = results
                    solved = True
                    break  # 当天得到可行解，停止放宽重试

            if not solved:
                # 当天放宽到最大仍不可行：跳过该日（或在此处记录日志）
                continue

        # 日循环结束后再聚合并返回
        selected_df = pd.concat(self.selected_list, axis=0, join="inner") if self.selected_list else pd.DataFrame(columns=['date','order_book_id','score','factor','industry','weight'])
        result_df = pd.DataFrame()
        result_df['date'] = selected_df['date']
        result_df['order_book_id'] = selected_df['order_book_id']
        result_df['score'] = selected_df[factor_name]
        result_df['factor'] = selected_df[factor_name]
        result_df['industry'] = selected_df['industry']
        result_df['weight'] = selected_df['weight']
        # result_df[['score', 'factor', 'weight', 'industry']] = (result_df.groupby('order_book_id')[['score', 'factor', 'weight', 'industry']].shift(1))
        result_df = result_df.dropna().set_index(['date', 'order_book_id'])
        return result_df

