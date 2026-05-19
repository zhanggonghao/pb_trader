import os

import numpy as np
import pandas as pd
import pulp

from ultis.stock_data_client import StockDataClient
# from stock_data_client import StockDataClient
# client = StockDataClient(data_path=r'//192.168.1.168/samba/Market/')
client = StockDataClient(data_path=r'//192.168.3.100/samba/Market/')
#
import rqdatac
# rqdatac.init(13601611030,'PB123456789')
# rqdatac.init(username='18101949790', password='123456')
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)

try:
    import cvxpy as cp
except ImportError:
    cp = None


def _log(message: str):
    print(f'[portfolio_optimizer] {message}', flush=True)


class Optimizer:
    def __init__(
        self,
        stock_data: pd.DataFrame,
        direction: int,
        style_exposures: pd.DataFrame = None,
        benchmark_style_exposures: pd.DataFrame = None,
        benchmark_industry_weights: pd.DataFrame = None,
        index_weight_limits: pd.DataFrame = None,
        stock_returns: pd.DataFrame = None,
        tracking_error_limit: float = None,
        vol_rolling_window: int = 60,
        bench_mark: str = '000300.XSHG',
    ):
        """
        初始化带权重分配的优化器。

        必要数据结构:
        - stock_data: index=['date','order_book_id']，列含:
            * factor_name: 用于目标函数（direction=1最大化，-1最小化）
            * 'industry': 股票所属行业

        可选:
        - style_exposures: index=['date','order_book_id']，多列风格暴露（*_exposure）
        - benchmark_style_exposures: index=['date']（或 MultiIndex 包含 'date'），列为风格名
        - benchmark_industry_weights: 若为 None，则按 bench_mark 内部用 client 拉取
        - index_weight_limits: index=['date','order_book_id']，列含 'weight' 和 'max_weight'，
            用于动态约束单票上限 (=min(指数权重*multiplier, fallback))
        - stock_returns: index=['date','order_book_id']，列含 'daily_return'，用于跟踪误差 QP
        - tracking_error_limit: 年化跟踪误差上限（启用 QP 求解，需要 stock_returns + index_weight_limits）
        - vol_rolling_window: 协方差矩阵滚动窗口
        """
        self.stock_data = stock_data.copy()

        if benchmark_industry_weights is None:
            start_date = self.stock_data.index.get_level_values('date').min()
            end_date = self.stock_data.index.get_level_values('date').max()
            benchmark_industry_weights = client.get_stock_index_comments_weights_industry(
                order_book_id=bench_mark, start=start_date, end=end_date,
            )
        self.benchmark_industry_weights = benchmark_industry_weights.copy()

        self.direction = direction
        self.selected_list = list()
        self.date_list = self.stock_data.index.get_level_values('date').unique().tolist()


        self.style_exposures = style_exposures
        self.benchmark_style_exposures = benchmark_style_exposures
        self.index_weight_limits = index_weight_limits
        self.stock_returns = stock_returns.sort_index() if stock_returns is not None else None
        self.tracking_error_limit = (
            None if tracking_error_limit is None else float(tracking_error_limit)
        )
        self.vol_rolling_window = int(vol_rolling_window)
        self.relax_records = list()

    def add_industry(self):
        self.stock_data = self.stock_data.reset_index()
        benchmark_industrys = client.get_all_instruments_industry_data().reset_index()
        self.stock_data = pd.merge(
            self.stock_data,
            benchmark_industrys[['date', 'order_book_id', 'industry']],
            on=['date', 'order_book_id'],
        ).set_index(['date', 'order_book_id'])
        print('推理文件添加行业', self.stock_data)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _parse_bounds(self, bounds):
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            return float(bounds[0]), float(bounds[1])
        raise ValueError(f"风格约束仅支持(lo, hi)或[lo, hi]，收到: {bounds}")

    def _get_day_weight_limits(self, trade_date):
        if self.index_weight_limits is None:
            return {}
        try:
            day_limits = self.index_weight_limits.xs(trade_date, level='date')
        except KeyError:
            return {}
        if 'max_weight' not in day_limits.columns:
            return {}
        return day_limits['max_weight'].astype(float).to_dict()

    def _get_day_benchmark_weights(self, trade_date):
        if self.index_weight_limits is None:
            return {}
        try:
            day_weights = self.index_weight_limits.xs(trade_date, level='date')
        except KeyError:
            return {}
        if 'weight' not in day_weights.columns:
            return {}
        return day_weights['weight'].astype(float).to_dict()

    def _get_benchmark_weights_vector(self, trade_date, stocks):
        bench_weights = self._get_day_benchmark_weights(trade_date)
        vec = np.array([bench_weights.get(stock, 0.0) for stock in stocks], dtype=float)
        total = float(vec.sum())
        if total > 0:
            vec = vec / total
        return vec

    def _nearest_psd(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, 1e-10, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _build_cov_matrix(self, trade_date, stocks):
        if self.stock_returns is None:
            return None
        if 'daily_return' not in self.stock_returns.columns:
            raise ValueError("stock_returns 缺少 daily_return 列")

        trade_date = pd.to_datetime(trade_date)
        hist = self.stock_returns[
            self.stock_returns.index.get_level_values('date') < trade_date
        ]
        if hist.empty:
            return None

        returns_panel = (
            hist['daily_return']
            .unstack('order_book_id')
            .sort_index()
            .reindex(columns=stocks)
            .tail(self.vol_rolling_window)
        )
        if len(returns_panel) < self.vol_rolling_window:
            return None

        cov_df = returns_panel.cov()
        cov_df = cov_df.reindex(index=stocks, columns=stocks)
        cov = cov_df.to_numpy(dtype=float)

        valid_diag = np.diag(cov)
        valid_diag = valid_diag[np.isfinite(valid_diag) & (valid_diag > 0)]
        diag_default = float(np.median(valid_diag)) if valid_diag.size > 0 else 1e-6

        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(len(stocks)):
            if cov[i, i] <= 0:
                cov[i, i] = diag_default

        cov += np.eye(len(stocks)) * 1e-8
        return self._nearest_psd(cov)

    def _build_style_sigma_df(self, style_sigma_bands):
        if (self.benchmark_style_exposures is None) or (style_sigma_bands is None):
            return None

        bench = self.benchmark_style_exposures.copy()
        if 'date' in getattr(bench, 'columns', []):
            bench['date'] = pd.to_datetime(bench['date'])
            bench = bench.set_index('date').sort_index()
        elif isinstance(bench.index, pd.MultiIndex) and 'date' in bench.index.names:
            bench = bench.reset_index().set_index('date').sort_index()
            bench = bench.drop(
                columns=[c for c in bench.columns if c in ['order_book_id', 'industry', 'weight']],
                errors='ignore',
            )
        else:
            bench.index = pd.to_datetime(bench.index)
            bench = bench.sort_index()

        bench = bench.apply(pd.to_numeric, errors='coerce')
        return bench.rolling(window=252, min_periods=252).std()

    def _build_industry_info(
        self,
        trade_date,
        stock_data,
        min_industry_limit,
        max_industry_limit,
        fixed_industry_limit,
    ):
        industry_info = dict()
        industry_weights = (
            self.benchmark_industry_weights.loc[trade_date]
            .groupby('industry')['weight'].sum().to_dict()
        )
        for industry, bench_weight in industry_weights.items():
            lo = bench_weight - bench_weight * min_industry_limit
            fixed_lo = bench_weight - fixed_industry_limit
            if fixed_lo > lo:
                lo = fixed_lo
            if lo < 0.01:
                lo = 0.0

            hi = bench_weight + bench_weight * max_industry_limit
            fixed_hi = bench_weight + fixed_industry_limit
            if fixed_hi < hi:
                hi = fixed_hi
            if hi < 0.01:
                hi = 0.01

            industry_stocks = stock_data[stock_data['industry'] == industry]['order_book_id'].tolist()
            if not industry_stocks:
                continue
            industry_info[industry] = {
                'lo': round(lo, 4),
                'hi': round(hi, 4),
                'stocks': industry_stocks,
            }
        return industry_info

    def _build_style_constraints(
        self,
        trade_date,
        stocks,
        sigma_df,
        style_sigma_bands,
        attempt,
        relax_multiplier,
    ):
        style_constraints = []
        eff_lo_sigma = None
        eff_hi_sigma = None
        used_bands = {}

        if (self.style_exposures is None) or (sigma_df is None) or (style_sigma_bands is None):
            return style_constraints, eff_lo_sigma, eff_hi_sigma, used_bands

        if trade_date not in self.style_exposures.index.get_level_values('date'):
            raise ValueError(f"{trade_date} 缺少风格暴露数据")

        exp_df = self.style_exposures.loc[trade_date].reindex(stocks).fillna(0.0)
        try:
            bench_row = self.benchmark_style_exposures.loc[trade_date]
            sigmas = sigma_df.loc[trade_date]
        except Exception:
            return style_constraints, eff_lo_sigma, eff_hi_sigma, used_bands

        if isinstance(bench_row, pd.DataFrame):
            bench_row = bench_row.iloc[0]
        if isinstance(sigmas, pd.DataFrame):
            sigmas = sigmas.iloc[0]

        if not isinstance(bench_row, pd.Series) or not isinstance(sigmas, pd.Series):
            return style_constraints, eff_lo_sigma, eff_hi_sigma, used_bands

        styles = [
            c for c in exp_df.columns
            if (c in bench_row.index) and (c in sigmas.index)
        ]
        scale = relax_multiplier ** attempt
        for style_name in styles:
            if isinstance(style_sigma_bands, dict):
                base_pair = style_sigma_bands.get(style_name, None)
            else:
                base_pair = style_sigma_bands
            if base_pair is None:
                continue

            base_lo, base_hi = self._parse_bounds(base_pair)
            lo_sigma = base_lo * scale
            hi_sigma = base_hi * scale
            eff_lo_sigma, eff_hi_sigma = lo_sigma, hi_sigma
            used_bands[style_name] = (lo_sigma, hi_sigma)

            bench_val = float(bench_row[style_name])
            sigma_val = float(sigmas[style_name])
            if pd.isna(sigma_val) or sigma_val <= 0:
                continue

            lo_abs = bench_val - lo_sigma * sigma_val
            hi_abs = bench_val + hi_sigma * sigma_val
            beta = exp_df[style_name].astype(float).to_numpy()
            style_constraints.append(
                {
                    'style_name': style_name,
                    'beta': beta,
                    'lo_abs': lo_abs,
                    'hi_abs': hi_abs,
                }
            )
            print(f'{style_name} 风格因子约束：{lo_abs}, {hi_abs}')
        return style_constraints, eff_lo_sigma, eff_hi_sigma, used_bands

    # ------------------------------------------------------------------
    # 单日 LP / QP 求解
    # ------------------------------------------------------------------
    def _solve_day_lp(
        self,
        stocks,
        scores,
        max_stock_weight,
        day_weight_limits,
        industry_info,
        style_constraints,
        use_turnover,
        prev_weights,
        turnover_limit,
    ):
        # print(f'stocks:{len(stocks)}, {stocks}')
        # print(f'scores:{len(scores)}, {scores}')
        # print(f'max_stock_weight:{max_stock_weight}')
        # print(f'day_weight_limits:{len(day_weight_limits)}, {day_weight_limits}')
        # print(f'industry_info:{len(industry_info)}, {industry_info}')
        # print(f'style_constraints:{len(style_constraints)}, {style_constraints}')
        # print(f'use_turnover: {use_turnover}')
        # print(f'prev_weights:{len(prev_weights)}, {prev_weights}')
        # print(f'turnover_limit:{turnover_limit}')

        prob = pulp.LpProblem(
            "IndustryOptimizer",
            pulp.LpMaximize if self.direction == 1 else pulp.LpMinimize,
        )


        weights = {}
        for stock in stocks:
            upper_bound = day_weight_limits.get(stock, max_stock_weight)
            weights[stock] = pulp.LpVariable(
                f"v_{stock}",
                lowBound=0,
                upBound=upper_bound,
                cat='Continuous',
            )

        prob += pulp.lpSum(weights[s] * scores[s] for s in stocks)
        prob += pulp.lpSum(weights[s] for s in stocks) == 1.0

        for ind, info in industry_info.items():
            ind_stocks = info['stocks']
            prob += pulp.lpSum(weights[s] for s in ind_stocks) >= info['lo']
            prob += pulp.lpSum(weights[s] for s in ind_stocks) <= info['hi']
            print(f'行业约束：{ind}，{info["lo"]}, {info["hi"]}')

        for style_constraint in style_constraints:
            beta = style_constraint['beta']
            comb = pulp.lpSum(weights[s] * beta[i] for i, s in enumerate(stocks))
            prob += comb >= style_constraint['lo_abs']
            prob += comb <= style_constraint['hi_abs']

        if use_turnover:
            z = {s: pulp.LpVariable(f"z_{s}", lowBound=0) for s in stocks}
            for s in stocks:
                old_w = prev_weights.get(s, 0.0)
                prob += weights[s] - old_w <= z[s]
                prob += old_w - weights[s] <= z[s]
            prob += pulp.lpSum(z.values()) <= 2 * turnover_limit

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            _log(
                f"[LP] PuLP 状态非 Optimal: status={status}, "
                f"n_stocks={len(stocks)}, use_turnover={use_turnover}"
            )
            return {}, status

        results = {s: weights[s].value() for s in stocks if weights[s].value() > 1e-5}
        return results, status

    def _solve_day_qp(
        self,
        trade_date,
        stocks,
        scores,
        max_stock_weight,
        day_weight_limits,
        industry_info,
        style_constraints,
        use_turnover,
        prev_weights,
        turnover_limit,
    ):
        if cp is None:
            raise ImportError("已启用波动率跟踪误差约束，但当前环境未安装 cvxpy")

        sigma = self._build_cov_matrix(trade_date, stocks)
        if sigma is None:
            _log(
                f"[QP] {trade_date} 无法构造协方差矩阵: "
                f"历史收益率不足或缺失 (需要至少 {self.vol_rolling_window} 个交易日), n_stocks={len(stocks)}"
            )
            return {}, 'InsufficientHistory'

        w_bench = self._get_benchmark_weights_vector(trade_date, stocks)
        if float(w_bench.sum()) <= 0:
            _log(
                f"[QP] {trade_date} 当日 index_weights 中无有效基准权重向量, "
                f"n_stocks={len(stocks)}"
            )
            return {}, 'MissingBenchmarkWeights'

        upper_bounds = np.array(
            [day_weight_limits.get(stock, max_stock_weight) for stock in stocks],
            dtype=float,
        )
        score_vector = np.array([scores[stock] for stock in stocks], dtype=float)
        te_limit_daily = float(self.tracking_error_limit) / np.sqrt(252.0)

        w = cp.Variable(len(stocks))
        constraints = [
            w >= 0,
            w <= upper_bounds,
            cp.sum(w) == 1.0,
        ]

        for info in industry_info.values():
            idx = [i for i, stock in enumerate(stocks) if stock in set(info['stocks'])]
            if not idx:
                continue
            constraints.append(cp.sum(w[idx]) >= info['lo'])
            constraints.append(cp.sum(w[idx]) <= info['hi'])

        for style_constraint in style_constraints:
            beta = style_constraint['beta']
            constraints.append(beta @ w >= style_constraint['lo_abs'])
            constraints.append(beta @ w <= style_constraint['hi_abs'])

        if use_turnover:
            prev_vector = np.array([prev_weights.get(stock, 0.0) for stock in stocks], dtype=float)
            z = cp.Variable(len(stocks), nonneg=True)
            constraints.append(w - prev_vector <= z)
            constraints.append(prev_vector - w <= z)
            constraints.append(cp.sum(z) <= 2 * turnover_limit)

        active_weights = w - w_bench
        constraints.append(cp.quad_form(active_weights, cp.psd_wrap(sigma)) <= te_limit_daily ** 2)

        objective = cp.Maximize(score_vector @ w) if self.direction == 1 else cp.Minimize(score_vector @ w)
        problem = cp.Problem(objective, constraints)

        solver_names = []
        installed = set(cp.installed_solvers())
        if 'ECOS' in installed:
            solver_names.append(cp.ECOS)
        if 'SCS' in installed:
            solver_names.append(cp.SCS)
        if not solver_names:
            solver_names = [None]

        for solver_name in solver_names:
            try:
                if solver_name is None:
                    problem.solve(verbose=False, warm_start=True)
                else:
                    problem.solve(solver=solver_name, verbose=False, warm_start=True)
            except Exception as exc:
                _log(f"[QP] {trade_date} 求解器异常: solver={solver_name}, err={exc!r}")
                continue

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                weight_values = np.asarray(w.value).reshape(-1)
                results = {
                    stock: float(weight_values[i])
                    for i, stock in enumerate(stocks)
                    if weight_values[i] is not None and weight_values[i] > 1e-5
                }
                return results, 'Optimal'
            _log(
                f"[QP] {trade_date} 求解未收敛: "
                f"solver={solver_name}, cvxpy_status={problem.status}, n_stocks={len(stocks)}"
            )

        _log(f"[QP] {trade_date} 所有求解器均失败, 最终 cvxpy_status={problem.status}")
        return {}, str(problem.status)

    # ------------------------------------------------------------------
    # 主优化循环
    # ------------------------------------------------------------------
    def optimize(
        self,
        factor_name,
        max_stock_weight=0.02,
        turnover_limit=0.15,
        min_industry_limit=0.2,
        max_industry_limit=0.5,
        fixed_industry_limit=0.02,
        style_sigma_bands=None,
        prev_weights=None,
    ):
        """
        :param style_sigma_bands:
            - (lo, hi): 全部风格统一的±Nσ带
            - dict: {style_name: (lo, hi)} 各风格单独±Nσ带
        :param prev_weights: 实盘场景由外部传入；回测场景一般在 main 中外部维护或传 None
        """
        self.relax_records = list()
        sigma_df = self._build_style_sigma_df(style_sigma_bands)
        relax_multiplier = 2.0
        max_relax_tries = 2
        use_tracking_qp = (
            self.stock_returns is not None and self.tracking_error_limit is not None
        )

        self.selected_list = []

        for di, trade_date in enumerate(self.date_list):
            print('正在处理日期：', trade_date)
            stock_data = self.stock_data.loc[trade_date].copy().reset_index()
            stock_data = stock_data[pd.notna(stock_data[factor_name])]
            if stock_data.empty:
                continue

            stocks = stock_data['order_book_id'].tolist()
            scores = dict(zip(stock_data['order_book_id'], stock_data[factor_name]))
            day_weight_limits = self._get_day_weight_limits(trade_date)
            industry_info = self._build_industry_info(
                trade_date=trade_date,
                stock_data=stock_data,
                min_industry_limit=min_industry_limit,
                max_industry_limit=max_industry_limit,
                fixed_industry_limit=fixed_industry_limit,
            )

            solved = False
            last_status = None
            for attempt in range(max_relax_tries + 1):
                style_constraints, eff_lo_sigma, eff_hi_sigma, used_bands = self._build_style_constraints(
                    trade_date=trade_date,
                    stocks=stocks,
                    sigma_df=sigma_df,
                    style_sigma_bands=style_sigma_bands,
                    attempt=attempt,
                    relax_multiplier=relax_multiplier,
                )

                # 回测多日：第一天 prev_weights=None，跳过 turnover
                # 实盘单日：prev_weights 由外部传入，只要非空就启用 turnover
                use_turnover = (
                    (prev_weights is not None)
                    and (len(prev_weights) > 0)
                    and (turnover_limit is not None)
                    and (turnover_limit > 0)
                )

                if use_tracking_qp:
                    results, status = self._solve_day_qp(
                        trade_date=trade_date,
                        stocks=stocks,
                        scores=scores,
                        max_stock_weight=max_stock_weight,
                        day_weight_limits=day_weight_limits,
                        industry_info=industry_info,
                        style_constraints=style_constraints,
                        use_turnover=use_turnover,
                        prev_weights=prev_weights,
                        turnover_limit=turnover_limit,
                    )
                else:
                    results, status = self._solve_day_lp(
                        stocks=stocks,
                        scores=scores,
                        max_stock_weight=max_stock_weight,
                        day_weight_limits=day_weight_limits,
                        industry_info=industry_info,
                        style_constraints=style_constraints,
                        use_turnover=use_turnover,
                        prev_weights=prev_weights,
                        turnover_limit=turnover_limit,
                    )

                self.relax_records.append(
                    {
                        'date': pd.to_datetime(trade_date),
                        'attempt': attempt,
                        'scale': float(relax_multiplier ** attempt),
                        'lo_sigma_used': eff_lo_sigma,
                        'hi_sigma_used': eff_hi_sigma,
                        'style_bands_used': used_bands,
                        'status': status,
                        'n_stocks': len(stocks),
                        'solver_mode': 'qp' if use_tracking_qp else 'lp',
                        'tracking_error_limit': self.tracking_error_limit,
                    }
                )
                last_status = status
                if status != 'Optimal':
                    _log(
                        f"{trade_date} attempt={attempt} "
                        f"scale={relax_multiplier ** attempt:.2f} "
                        f"mode={'QP' if use_tracking_qp else 'LP'} status={status} n_stocks={len(stocks)}"
                    )

                if status == 'Optimal':
                    sel = stock_data[stock_data['order_book_id'].isin(results.keys())].copy()
                    sel['date'] = trade_date
                    sel['weight'] = sel['order_book_id'].map(results)
                    sel = sel.sort_values(by='industry', ascending=False)
                    self.selected_list.append(sel)
                    prev_weights = results
                    solved = True
                    break

            if not solved:
                print(
                    f"[portfolio_optimizer] 跳过日期 {pd.to_datetime(trade_date).date()}: "
                    f"经过 {max_relax_tries + 1} 次尝试仍无可行解, 最后状态={last_status}, "
                    f"n_stocks={len(stocks)}, mode={'QP' if use_tracking_qp else 'LP'}, "
                    f"use_turnover={use_turnover}"
                )
                continue

        selected_df = (
            pd.concat(self.selected_list, axis=0, join="inner")
            if self.selected_list
            else pd.DataFrame(columns=['date', 'order_book_id', 'score', 'factor', 'industry', 'weight'])
        )
        n_solved_days = len(self.selected_list)
        n_all_days = len(self.date_list)
        if n_solved_days == 0:
            _log(f"优化结束: 无任何交易日得到可行解 (共遍历 {n_all_days} 个日期)")
            print(f"[portfolio_optimizer] 警告: 全部 {n_all_days} 个交易日均未得到可行解, 返回空持仓表")
        else:
            skipped = n_all_days - n_solved_days
            if skipped > 0:
                print(
                    f"[portfolio_optimizer] 汇总: 成功优化 {n_solved_days} 个交易日, "
                    f"因无解跳过约 {skipped} 个因子日 (以 relax_records 为准)"
                )
            _log(f"优化结束: 成功 {n_solved_days} 日 / 因子日共 {n_all_days} 日")

        # print(selected_df)
        result_df = pd.DataFrame()
        result_df['date'] = selected_df['date']
        result_df['order_book_id'] = selected_df['order_book_id']
        result_df['score'] = selected_df[factor_name]
        result_df['factor'] = selected_df[factor_name]
        result_df['industry'] = selected_df['industry']
        result_df['weight'] = selected_df['weight']
        result_df = result_df.dropna().set_index(['date', 'order_book_id'])
        return result_df


# ----------------------------------------------------------------------
# IO 工具
# ----------------------------------------------------------------------
def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _ensure_mi(df: pd.DataFrame) -> pd.DataFrame:
    if not {'date', 'order_book_id'}.issubset(df.columns):
        if isinstance(df.index, pd.MultiIndex) and set(df.index.names) == {'date', 'order_book_id'}:
            return df
        raise ValueError("表需包含列: date, order_book_id 或 MultiIndex ['date','order_book_id']")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index(['date', 'order_book_id']).sort_index()


def _save_table(df: pd.DataFrame, out: str):
    ext = os.path.splitext(out)[1].lower()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    if ext == '.parquet':
        df.to_parquet(out)
    elif ext in ['.csv', '.txt']:
        df.reset_index().to_csv(out, index=False)
    elif ext in ['.feather', '.ft']:
        df.reset_index().to_feather(out)
    else:
        df.to_parquet(out)



def _normalize_index_weight_limits(
    index_weight_limits,
    multiplier: float,
    fallback: float,
    stock_weight_cap: float = None,
    start_date=None,
    end_date=None,
):
    """
    规范化外部传入的 index_weight_limits（来自 tranform_target.py.get_index_weight_limits）：
    - 接受 MultiIndex(date, order_book_id) 的 DataFrame，或含 ['date','order_book_id'] 列的长表；
    - 必须含 'weight'；若缺 'max_weight'，按 np.maximum(weight*multiplier, fallback) 现场补齐；
    - 若 stock_weight_cap 非 None，则无论是否已有 'max_weight'，都会再限制 max_weight <= stock_weight_cap；
    - 按 [start_date, end_date] 过滤。
    """
    if index_weight_limits is None:
        _log('未传入 index_weight_limits，单票上限回退为固定权重上限')
        return None

    df = index_weight_limits.copy()
    if isinstance(df.index, pd.MultiIndex) and set(df.index.names) >= {'date', 'order_book_id'}:
        df = df.reset_index()

    required_cols = {'date', 'order_book_id', 'weight'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"index_weight_limits 缺少必要列 {required_cols}，实际列为: {list(df.columns)}"
        )

    df['date'] = pd.to_datetime(df['date'])
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df.dropna(subset=['date', 'order_book_id', 'weight'])

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    if 'max_weight' not in df.columns:
        df['max_weight'] = np.maximum(
            df['weight'].astype(float) * float(multiplier),
            float(fallback),
        )
    else:
        df['max_weight'] = pd.to_numeric(df['max_weight'], errors='coerce')

    if stock_weight_cap is not None:
        df['max_weight'] = np.minimum(df['max_weight'], float(stock_weight_cap))

    df = df[df['max_weight'] > 0]
    df = df.drop_duplicates(subset=['date', 'order_book_id'], keep='last')
    df = df.set_index(['date', 'order_book_id']).sort_index()
    _log(
        f"已应用动态单票权重上限: 行数:{len(df)}, "
        f"规则: max_weight=np.maximum(weight*{multiplier:.3f}, {fallback:.4f})"
        f"{'' if stock_weight_cap is None else f' capped_by={float(stock_weight_cap):.4f}'}"
    )
    return df


def _load_stock_returns(stock_returns_path: str, end_date=None):
    """
    加载本地 stock_returns 文件，要求 MultiIndex(date, order_book_id) 含 'daily_return' 列。
    """
    if not stock_returns_path:
        return None
    if not os.path.exists(stock_returns_path):
        _log(f'未找到收益率文件: {stock_returns_path}，跟踪误差约束将被忽略')
        return None

    ext = os.path.splitext(stock_returns_path)[1].lower()
    if ext in ['.feather', '.ft']:
        df = pd.read_feather(stock_returns_path)
    elif ext in ['.csv', '.txt']:
        df = pd.read_csv(stock_returns_path)
    else:
        df = pd.read_parquet(stock_returns_path)

    if {'date', 'order_book_id'}.issubset(df.columns):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'order_book_id']).sort_index()
    elif not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "stock_returns 需为 MultiIndex(date, order_book_id) 或含 ['date','order_book_id'] 列"
        )
    if 'daily_return' not in df.columns:
        raise ValueError("stock_returns 缺少 daily_return 列")
    if end_date is not None:
        df = df[df.index.get_level_values('date') <= pd.to_datetime(end_date)]
    _log(f'已加载个股收益率文件: {stock_returns_path}, 行数:{len(df)}')
    return df


# ----------------------------------------------------------------------
# 对外 main 入口（被 tranform_target.py 调用）
# ----------------------------------------------------------------------
def main(
    trading_params, stock_df, prev_weights,
    style_df=None, bench_style_df=None,
    index_weight_limits=None,
):
    """
    实盘对接入口（向后兼容；新增可选 index_weight_limits）：
        portfolio_optimizer_re.main(
            trading_params,           # 单 unique_id 的参数字典（来自 yaml）
            stock_df,                 # 含 ['order_book_id','date','score','industry']
            prev_weights,             # dict 或 None，上一交易日已落地的目标持仓权重
            style_df=stock_factors,   # 个股 *_exposure 表（MultiIndex）
            bench_style_df=index_exposures,  # 指数 *_exposure 表（含 'date' 列）
            index_weight_limits=...,  # 由外部本地按 generate_index_weights.py 算法预先算好，
                                      # MultiIndex(date, order_book_id) + ['weight','max_weight']
        )

    返回: result_df，index=['date','order_book_id']，列含 ['score','factor','industry','weight']
    """
    fname = trading_params['factor_name']

    # 防止外部传入的 DataFrame 被就地修改
    stock_df = stock_df.copy()
    if style_df is not None:
        style_df = style_df.copy()
    if bench_style_df is not None:
        bench_style_df = bench_style_df.copy()

    # ----------- 规范化 stock_df: MultiIndex + 'score' 重命名 -----------
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df = (
        stock_df.set_index(['date', 'order_book_id']).sort_index()
                .rename(columns={'score': fname})
    )

    # ----------- benchmark_industry_weights 基准行业权重 -----------
    bench_mark = trading_params.get('bench_mark', '000300.XSHG')
    bench_df = client.get_stock_index_comments_weights_industry(
        order_book_id=bench_mark,
        start=stock_df.index.get_level_values('date').min(),
        end=stock_df.index.get_level_values('date').max(),
    )

    # ----------- bench_style_df 索引规范 -----------
    if bench_style_df is not None:
        if 'date' in bench_style_df.columns:
            bench_style_df['date'] = pd.to_datetime(bench_style_df['date'])
            bench_style_df = bench_style_df.set_index('date').sort_index()
        else:
            if isinstance(bench_style_df.index, pd.MultiIndex) and 'date' in bench_style_df.index.names:
                pass
            else:
                raise ValueError("benchmark-style-exposures 需含 'date'（列或索引层）")

    # ============= T -> T+1 平移（与 backtest 同款）=============
    # 1) 个股风格暴露：仅平移以 _exposure 结尾的列
    if style_df is not None:
        exp_cols = [c for c in style_df.columns if str(c).endswith('_exposure')]
        if exp_cols:
            _pu = style_df[exp_cols].unstack('order_book_id')  # index: date
            # style_df[exp_cols] = _pu.shift(1).stack('order_book_id').reindex(style_df.index)
            style_df[exp_cols] = _pu.stack('order_book_id').reindex(style_df.index)

    # 2) 指数风格暴露：整表按日平移
    if bench_style_df is not None:
        # bench_style_df = bench_style_df.sort_index().shift(1)
        bench_style_df = bench_style_df.sort_index()
    # ===========================================================

    # ----------- 风格 σ 带宽 -----------
    style_sigma_bands = trading_params.get('style_index_bands_sigma', None)
    lo = trading_params.get('style_index_bands_sigma_lo', None)
    hi = trading_params.get('style_index_bands_sigma_hi', None)
    if style_sigma_bands is None and (lo is not None and hi is not None):
        style_sigma_bands = (lo, hi)

    # ----------- 日期对齐（与 backtest 一致：仅对齐风格表） -----------
    common_dates = stock_df.index.get_level_values('date').unique().intersection(
        bench_df.index.get_level_values('date').unique()
    )
    if style_df is not None:
        common_dates = common_dates.intersection(style_df.index.get_level_values('date').unique())
    if bench_style_df is not None:
        bs_dates = (
            bench_style_df.index.get_level_values('date')
            if isinstance(bench_style_df.index, pd.MultiIndex)
            else bench_style_df.index
        )
        common_dates = common_dates.intersection(pd.to_datetime(bs_dates).unique())
    if style_df is not None:
        style_df = style_df.loc[style_df.index.get_level_values('date').isin(common_dates)]

    # ----------- 动态单票上限 / 跟踪误差所需数据 -----------
    stock_weight_multiplier = float(trading_params.get('stock_weight_multiplier', 1.1))
    stock_weight_fallback = float(
        trading_params.get('stock_weight_fallback', trading_params.get('max_stock_weight', 0.02))
    )
    stock_weight_cap = trading_params.get('stock_weight_cap', None)
    stock_weight_cap = None if stock_weight_cap is None else float(stock_weight_cap)
    max_stock_weight = float(trading_params['max_stock_weight'])
    if stock_weight_cap is not None:
        max_stock_weight = min(max_stock_weight, stock_weight_cap)
    fixed_industry_limit = float(trading_params.get('fixed_industry_limit', 0.02))
    stock_returns_path = trading_params.get('stock_returns_path', None)
    tracking_error_limit = trading_params.get('tracking_error_limit', None)
    vol_rolling_window = int(trading_params.get('vol_rolling_window', 60))

    start_date = stock_df.index.get_level_values('date').min()
    end_date = stock_df.index.get_level_values('date').max()
    # 个股基准权重 + 动态单票上限：由外部（tranform_target.py）按 generate_index_weights.py 的
    # 算法在本地预先算好，再通过参数传入；缺失时退化为固定上限。
    index_weight_limits = _normalize_index_weight_limits(
        index_weight_limits=index_weight_limits,
        multiplier=stock_weight_multiplier,
        fallback=stock_weight_fallback,
        stock_weight_cap=stock_weight_cap,
        start_date=start_date,
        end_date=end_date,
    )
    stock_returns = _load_stock_returns(stock_returns_path=stock_returns_path, end_date=end_date)
    if tracking_error_limit is not None and stock_returns is None:
        raise ValueError('已配置 tracking_error_limit，但未能加载 stock_returns_path 对应的收益率数据')
    if tracking_error_limit is not None and index_weight_limits is None:
        raise ValueError(
            '已配置 tracking_error_limit，但未传入 index_weight_limits（需要基准权重向量构造跟踪误差）'
        )

    # ----------- 优化 -----------

    # print(f'index_weight_limits:{index_weight_limits}')
    # print(f'style_df:{style_df}')
    # print(f'bench_style_df:{bench_style_df}')
    # print(f'bench_df:{bench_df}')

    opt = Optimizer(
        stock_data=stock_df,
        direction=int(trading_params['direction']),
        style_exposures=style_df,
        benchmark_style_exposures=bench_style_df,
        benchmark_industry_weights=bench_df,
        index_weight_limits=index_weight_limits,
        stock_returns=stock_returns,
        tracking_error_limit=tracking_error_limit,
        vol_rolling_window=vol_rolling_window,
        bench_mark=bench_mark,
    )


    res = opt.optimize(
        factor_name=fname,
        max_stock_weight=trading_params['max_stock_weight'],
        turnover_limit=trading_params['turnover_limit'],
        min_industry_limit=trading_params['min_industry'],
        max_industry_limit=trading_params['max_industry'],
        fixed_industry_limit=fixed_industry_limit,
        style_sigma_bands=style_sigma_bands,
        prev_weights=prev_weights,
    )

    res = res.sort_index()

    relax_df = pd.DataFrame(opt.relax_records)
    if not relax_df.empty:
        print(f'放宽日志: {relax_df}')

    return res


if __name__ == '__main__':
    paths = {
        'stock_data': '/home/trading/pred_df/df_test_PB_ScorpioV4_2020_20250919.parquet',
        'style_exposures': '/home/zhanggh/TransformTargetData/factors/906_exposures.parquet',
        'benchmark_style_exposures': '/home/zhanggh/TransformTargetData/factors/300_index_exposures.parquet',
        'out': r"E:\alphagen-master\alpha_test\optimizer\results\weights_0909.parquet",
    }
    trading_params = {
        'factor_name': 'avg_rank_4',
        'direction': 1,
        'max_stock_weight': 0.0125,
        'turnover_limit': 0.15,
        'min_industry': 0.4,
        'max_industry': 1.0,
        'fixed_industry_limit': 0.02,
        'bench_mark': '000300.XSHG',
        'stock_weight_multiplier': 1.1,
        'stock_weight_fallback': 0.0125,
        'style_index_bands_sigma': {
            'lcap_exposure': (0.5, 0.5),
            'beta_exposure': (1.0, 1.0),
            'liquidity_exposure': (0.5, 0.5),
        },
    }
    main(trading_params, _read_parquet(paths['stock_data']).reset_index(), None,
         style_df=_read_parquet(paths['style_exposures']),
         bench_style_df=_read_parquet(paths['benchmark_style_exposures']))
