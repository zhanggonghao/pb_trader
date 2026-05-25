"""
计算模块 - 实现所有模块的计算逻辑

本模块包含所有报告指标的核心计算逻辑：

**模块1：周期净值信息**
- 计算净值增长率、基准收益率、最大回撤等

**模块2：每日收益明细**
- 计算每日收益率及贡献分解

**模块3：多空详情**
- 股票组合收益、期货收益、基差收益分解

**模块4：每日个股权重Top10**
- 按市值权重排序，取前10名

**模块5：持仓集中度**
- 计算前20%重仓股的累计权重

**模块6：因子风格敞口**
- 计算组合和基准在各因子上的暴露

所有计算参数均从config模块读取。
"""
import pandas as pd
import numpy as np
from datetime import datetime

import config


class ReportCalculator:
    def __init__(self, daily_data, benchmark_prices, stock_prices=None, style_factors=None, benchmark_factors=None):
        self.daily_data = daily_data
        self.benchmark_prices = benchmark_prices
        self.stock_prices = stock_prices
        self.style_factors = style_factors
        self.benchmark_factors = benchmark_factors
        self.report_dates = daily_data['dates']
        self.prev_date = daily_data['prev_date']
        self.accounts = daily_data['accounts']

    def get_nav(self, date_str):
        """获取净值"""
        return self.daily_data['nav'].get(date_str)

    def get_stk_net_assets(self, date_str, account):
        """获取证券户净资产"""
        info = self.daily_data['stk_accounts'].get(date_str, {}).get(account)
        if info is not None:
            cols = list(info.index)
            if '证券户净资产' in cols:
                return float(info['证券户净资产'])
        return None

    def get_stk_long_mv(self, date_str, account):
        """获取证券户多头持仓市值"""
        info = self.daily_data['stk_accounts'].get(date_str, {}).get(account)
        if info is not None:
            cols = list(info.index)
            if '多头持仓市值' in cols:
                return float(info['多头持仓市值'])
        return None

    def get_stk_long_pnl(self, date_str, account):
        """获取证券户多头总收益"""
        info = self.daily_data['stk_accounts'].get(date_str, {}).get(account)
        if info is not None:
            cols = list(info.index)
            if '多头总收益' in cols:
                return float(info['多头总收益'])
        return None

    def get_fut_static_rights(self, date_str):
        """获取期货静态权益"""
        info = self.daily_data['fut'].get(date_str)
        if info is not None:
            cols = list(info.index)
            if '期货户静态权益' in cols:
                return float(info['期货户静态权益'])
        return None

    def get_fut_margin(self, date_str):
        """获取期货保证金"""
        info = self.daily_data['fut'].get(date_str)
        if info is not None:
            cols = list(info.index)
            if '期货户保证金' in cols:
                return float(info['期货户保证金'])
        return None

    def get_fut_pnl(self, date_str):
        """获取期货收益"""
        info = self.daily_data['fut'].get(date_str)
        if info is not None:
            cols = list(info.index)
            if '期货户收益' in cols:
                return float(info['期货户收益'])
        return None

    def get_product_total_assets(self, date_str):
        """计算产品总净资产"""
        total = 0
        for acc in self.accounts:
            net = self.get_stk_net_assets(date_str, acc)
            if net is not None:
                total += net
        fut = self.get_fut_static_rights(date_str)
        if fut is not None:
            total += fut
        return total if total > 0 else None

    def get_benchmark_return(self, date_str):
        """计算基准日收益率"""
        prev = self._prev_date_str(date_str)
        curr_price = self.benchmark_prices.get(date_str)
        prev_price = self.benchmark_prices.get(prev)
        if curr_price and prev_price and prev_price != 0:
            return (curr_price / prev_price) - 1
        return None

    def _prev_date_str(self, date_str):
        """获取前一交易日"""
        dates_with_prev = [self.prev_date] + self.report_dates
        idx = dates_with_prev.index(date_str)
        if idx > 0:
            return dates_with_prev[idx - 1]
        return None

    def calc_module1(self):
        """模块1：周期净值信息"""
        end_nav = self.get_nav(self.report_dates[-1])
        start_nav = self.get_nav(self.prev_date)

        end_date_str = self.report_dates[-1]
        start_date_str = self.prev_date
        end_bench = self.benchmark_prices.get(end_date_str)
        start_bench = self.benchmark_prices.get(start_date_str)

        nav_growth = (end_nav / start_nav - 1) if (start_nav and end_nav) else None
        bench_growth = (end_bench / start_bench - 1) if (start_bench and end_bench) else None

        nav_curve = {}
        bench_curve = {}
        nav_values = []

        dates_with_prev = [self.prev_date] + self.report_dates
        for d in dates_with_prev:
            nav = self.get_nav(d)
            bench = self.benchmark_prices.get(d)
            if nav:
                nav_curve[d] = nav
                nav_values.append(nav)
            if bench:
                bench_curve[d] = bench

        # 正确的最大回撤计算：peak_value初始化为第一个净值，遍历时动态更新
        max_drawdown = 0
        if nav_values:
            peak_value = nav_values[0]  # 初始化为第一个净值
            for nav in nav_values:
                if nav > peak_value:
                    peak_value = nav  # 更新历史峰值
                if peak_value > 0:
                    drawdown = (peak_value - nav) / peak_value
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

        return {
            'nav_growth': nav_growth,
            'bench_growth': bench_growth,
            'max_drawdown': max_drawdown,
            'nav_curve': nav_curve,
            'bench_curve': bench_curve,
            'nav_dates': dates_with_prev,
        }

    def calc_module2(self):
        """模块2：每日多头组合收益率、基准收益率、超额"""
        results = []

        for d in self.report_dates:
            total_long_pnl = 0
            total_net_assets = 0

            for acc in self.accounts:
                pnl = self.get_stk_long_pnl(d, acc)
                net = self.get_stk_net_assets(d, acc)
                if pnl is not None and net is not None and net != 0:
                    total_long_pnl += pnl
                    total_net_assets += net

            r_stk = (total_long_pnl / total_net_assets) if total_net_assets != 0 else None
            r_bench = self.get_benchmark_return(d)
            excess = (r_stk - r_bench) if (r_stk is not None and r_bench is not None) else None

            results.append({
                'date': d,
                'stk_return': r_stk,
                'bench_return': r_bench,
                'excess': excess,
                'total_pnl': total_long_pnl,
                'total_net_assets': total_net_assets,
            })

        return results

    def calc_module3(self):
        """模块3：多头空头详细信息"""
        results = []
        dates_with_prev = [self.prev_date] + self.report_dates

        for i, d in enumerate(self.report_dates):
            prev_d = dates_with_prev[i]

            total_long_pnl = 0
            total_net_assets = 0
            total_long_mv = 0

            for acc in self.accounts:
                pnl = self.get_stk_long_pnl(d, acc)
                net = self.get_stk_net_assets(d, acc)
                mv = self.get_stk_long_mv(d, acc)
                if pnl is not None:
                    total_long_pnl += pnl
                if net is not None:
                    total_net_assets += net
                if mv is not None:
                    total_long_mv += mv

            fut_pnl = self.get_fut_pnl(d)
            fut_margin = self.get_fut_margin(d)
            # 使用配置中的期货保证金比例计算名义市值
            margin_ratio = config.CalculationParams.FUTURE_MARGIN_RATIO
            fut_nominal_mv = (fut_margin / margin_ratio) if fut_margin else None

            r_bench = self.get_benchmark_return(d)

            r_stk = (total_long_pnl / total_net_assets) if total_net_assets != 0 else None
            r_fut = (fut_pnl / fut_nominal_mv) if (fut_pnl is not None and fut_nominal_mv and fut_nominal_mv != 0) else None
            basis_return = (r_fut + r_bench) if (r_fut is not None and r_bench is not None) else None

            product_assets = self.get_product_total_assets(d)

            stk_contrib = None
            fut_contrib = None
            if product_assets and product_assets > 0:
                if total_long_pnl is not None and total_long_mv is not None and r_bench is not None:
                    stk_contrib = (total_long_pnl - total_long_mv * r_bench) / product_assets
                if fut_pnl is not None and fut_nominal_mv is not None and r_bench is not None:
                    fut_contrib = (fut_pnl + fut_nominal_mv * r_bench) / product_assets

            total_contrib = (stk_contrib + fut_contrib) if (stk_contrib is not None and fut_contrib is not None) else None

            account_details = []
            for acc in self.accounts:
                pnl = self.get_stk_long_pnl(d, acc)
                net = self.get_stk_net_assets(d, acc)
                r_acc = (pnl / net) if (pnl is not None and net is not None and net != 0) else None
                excess_acc = (r_acc - r_bench) if (r_acc is not None and r_bench is not None) else None
                account_details.append({
                    'account': acc,
                    'return': r_acc,
                    'excess': excess_acc,
                    'pnl': pnl,
                    'net_assets': net,
                })

            results.append({
                'date': d,
                'accounts': account_details,
                'total_stk_return': r_stk,
                'fut_return': r_fut,
                'basis_return': basis_return,
                'stk_contrib': stk_contrib,
                'fut_contrib': fut_contrib,
                'total_contrib': total_contrib,
                'product_assets': product_assets,
            })

        return results

    def calc_module4(self, positions_data, stock_prices):
        """模块4：每日个股权重Top10"""
        results = {}

        for d in self.report_dates:
            if d not in positions_data or positions_data[d] is None:
                continue

            pos_df = positions_data[d].copy()

            if stock_prices is None:
                stock_prices = {}

            pos_df['code_full'] = pos_df['code'].apply(self._convert_stock_code)

            merged_df = pos_df.groupby('code_full', as_index=False)['hold'].sum()

            merged_df['market_value'] = 0.0
            for idx, row in merged_df.iterrows():
                code = row['code_full']
                hold = row['hold']
                if isinstance(stock_prices, dict) and code in stock_prices and isinstance(stock_prices[code], dict) and d in stock_prices[code]:
                    price = stock_prices[code][d]
                    if price and price > 0:
                        merged_df.at[idx, 'market_value'] = hold * price
                    else:
                        merged_df.at[idx, 'market_value'] = 0
                else:
                    merged_df.at[idx, 'market_value'] = 0

            total_mv = merged_df['market_value'].sum()
            if total_mv > 0:
                merged_df['weight'] = merged_df['market_value'] / total_mv
            else:
                merged_df['weight'] = 0

            pos_df_sorted = merged_df.sort_values('weight', ascending=False)
            top10 = pos_df_sorted.head(10)

            top10_records = []
            for _, row in top10.iterrows():
                top10_records.append({
                    'code': row['code_full'],
                    'hold': row['hold'],
                    'market_value': row['market_value'],
                    'weight': row['weight'],
                })

            results[d] = {
                'top10': top10_records,
                'total_mv': total_mv,
            }

        return results

    def _convert_stock_code(self, code):
        """
        转换股票代码格式
        
        根据配置中的股票代码规则进行转换：
        - 60开头 → XSHG（沪市）
        - 688开头 → XSHG（科创板）
        - 其他 → XSHE（深市）
        
        参数:
            code: 股票代码（可以是整数、字符串，带或不带后缀）
        
        返回:
            str: 完整的股票代码，如 '600000.XSHG'
        """
        code_str = str(code)
        if '.' in code_str:
            return code_str
        
        code_padded = code_str.zfill(6)
        
        # 根据配置的规则判断市场
        for market, prefixes in config.CalculationParams.STOCK_CODE_RULES.items():
            for prefix in prefixes:
                if code_padded.startswith(prefix):
                    if market == 'SH':
                        return f"{code_padded}.XSHG"
                    else:
                        return f"{code_padded}.XSHE"
        
        # 默认返回深市
        return f"{code_padded}.XSHE"

    def calc_module5(self, positions_data, stock_prices):
        """模块5：持仓集中度"""
        results = {}

        for d in self.report_dates:
            if d not in positions_data or positions_data[d] is None:
                continue

            pos_df = positions_data[d].copy()

            if stock_prices is None:
                stock_prices = {}

            pos_df['code_full'] = pos_df['code'].apply(self._convert_stock_code)

            merged_df = pos_df.groupby('code_full', as_index=False)['hold'].sum()

            merged_df['market_value'] = 0.0
            for idx, row in merged_df.iterrows():
                code = row['code_full']
                hold = row['hold']
                if isinstance(stock_prices, dict) and code in stock_prices and isinstance(stock_prices[code], dict) and d in stock_prices[code]:
                    price = stock_prices[code][d]
                    if price and price > 0:
                        merged_df.at[idx, 'market_value'] = hold * price
                else:
                    merged_df.at[idx, 'market_value'] = 0

            total_mv = merged_df['market_value'].sum()
            if total_mv <= 0:
                continue

            merged_df['weight'] = merged_df['market_value'] / total_mv
            pos_df_sorted = merged_df.sort_values('weight', ascending=False).reset_index(drop=True)
            pos_df_sorted['cum_weight'] = pos_df_sorted['weight'].cumsum()

            total_stocks = len(pos_df_sorted)
            # 使用配置中的集中度计算比例
            top_pct = config.CalculationParams.CONCENTRATION_TOP_PCT
            top_pct_count = max(1, int(total_stocks * top_pct))
            top_pct_weight = pos_df_sorted.iloc[top_pct_count - 1]['cum_weight'] if top_pct_count > 0 else 0

            results[d] = {
                'total_stocks': total_stocks,
                'top_20_pct_count': top_pct_count,
                'top_20_pct_weight': top_pct_weight,
                'all_weights': pos_df_sorted[['code_full', 'weight', 'cum_weight']].to_dict('records'),
            }

        return results

    def calc_module6(self, positions_data, stock_prices, factor_exposure_data=None):
        """
        模块6：因子风格敞口
        
        使用本地parquet文件中的因子暴露数据计算组合和基准的因子敞口
        
        参数:
            positions_data: 持仓数据 {date: DataFrame}
            stock_prices: 股票价格数据（用于计算市值权重）
            factor_exposure_data: 预计算的因子暴露数据 {date: {'portfolio': {...}, 'benchmark': {...}}}
        """
        results = {}
        
        # 如果传入了预计算的因子数据，直接使用
        if factor_exposure_data:
            return factor_exposure_data
        
        # 否则使用旧的米筐数据（兼容旧逻辑）
        if not self.style_factors or not self.benchmark_factors:
            return results

        for d in self.report_dates:
            if d not in positions_data or positions_data[d] is None:
                continue

            pos_df = positions_data[d].copy()

            if stock_prices is None:
                stock_prices = {}

            pos_df['code_full'] = pos_df['code'].apply(self._convert_stock_code)

            merged_df = pos_df.groupby('code_full', as_index=False)['hold'].sum()

            merged_df['market_value'] = 0.0
            for idx, row in merged_df.iterrows():
                code = row['code_full']
                hold = row['hold']
                if isinstance(stock_prices, dict) and code in stock_prices and isinstance(stock_prices[code], dict) and d in stock_prices[code]:
                    price = stock_prices[code][d]
                    if price and price > 0:
                        merged_df.at[idx, 'market_value'] = hold * price

            total_mv = merged_df['market_value'].sum()
            if total_mv <= 0:
                continue

            merged_df['weight'] = merged_df['market_value'] / total_mv
            # 使用配置中的因子列表
            factors = config.CalculationParams.FACTOR_LIST


            portfolio_factors = {}
            for factor in factors:
                weighted_sum = 0
                count = 0
                for idx, row in merged_df.iterrows():
                    code = row['code_full']
                    weight = row['weight']
                    if self.style_factors and d in self.style_factors and code in self.style_factors[d].get(factor, {}):
                        factor_value = self.style_factors[d][factor][code]
                        if factor_value is not None and not np.isnan(factor_value):
                            weighted_sum += factor_value * weight
                            count += 1
                if count > 0:
                    portfolio_factors[factor] = weighted_sum
                else:
                    portfolio_factors[factor] = None

            results[d] = {
                'portfolio': portfolio_factors,
                'benchmark': self.benchmark_factors.get(d, {}),
            }

        return results

    def calc_module7(self, positions_data, stock_prices, industry_data):
        """
        模块7：行业对比（申万一级行业）
        
        计算组合持仓和基准（沪深300）在各申万一级行业的权重分布，
        用于对比分析行业偏离度。
        
        参数:
            positions_data: 持仓数据 {date: DataFrame}
            stock_prices: 股票价格数据
            industry_data: 行业数据，包含组合和基准的行业分类
                {date: {code: industry_code}, f'{date}_bench': {industry_code: weight}}
        """
        results = {}

        for d in self.report_dates:
            if d not in positions_data or positions_data[d] is None:
                continue

            pos_df = positions_data[d].copy()

            if stock_prices is None:
                stock_prices = {}

            pos_df['code_full'] = pos_df['code'].apply(self._convert_stock_code)
            merged_df = pos_df.groupby('code_full', as_index=False)['hold'].sum()

            # 计算市值
            merged_df['market_value'] = 0.0
            for idx, row in merged_df.iterrows():
                code = row['code_full']
                hold = row['hold']
                if isinstance(stock_prices, dict) and code in stock_prices and isinstance(stock_prices[code], dict) and d in stock_prices[code]:
                    price = stock_prices[code][d]
                    if price and price > 0:
                        merged_df.at[idx, 'market_value'] = hold * price

            total_mv = merged_df['market_value'].sum()
            if total_mv <= 0:
                continue

            merged_df['weight'] = merged_df['market_value'] / total_mv

            # 获取组合行业分类
            portfolio_industry_map = industry_data.get(d, {})
            benchmark_industry_weights = industry_data.get(f'{d}_bench', {})

            if not portfolio_industry_map:
                continue

            # 计算组合各行业权重
            portfolio_industry = {}
            for idx, row in merged_df.iterrows():
                code = row['code_full']
                weight = row['weight']
                industry = portfolio_industry_map.get(code)
                if industry:
                    portfolio_industry[industry] = portfolio_industry.get(industry, 0) + weight

            # 基准行业权重直接使用（已经是行业名称维度）
            benchmark_industry = dict(benchmark_industry_weights)

            results[d] = {
                'portfolio_industry': portfolio_industry,
                'benchmark_industry': benchmark_industry,
            }

        return results

    def calc_market_review_data(self, market_data, module3_data):
        """
        计算市场回顾所需数据
        
        参数:
            market_data: 市场宏观数据（来自rq_fetcher.get_market_overview_data）
            module3_data: 模块3数据（多空详情，包含贡献分解）
        
        返回:
            dict: 市场回顾指标
        """
        print('market_data')
        print(market_data)
        if not market_data:
            return None

        # print('market_data')
        # print(market_data)

        result = {}

        # 获取日期
        dates = self.report_dates
        prev_date = self.prev_date
        end_date = dates[-1]

        # 1. 计算指数周涨跌幅
        def calc_return(data_dict, start_d, end_d):
            if not data_dict or start_d not in data_dict or end_d not in data_dict:
                return None
            start_val = data_dict[start_d]
            end_val = data_dict[end_d]
            if isinstance(start_val, dict):
                start_val = start_val['close']
                end_val = end_val['close']
            return (end_val / start_val - 1) if start_val else None
        result['sh_index_return'] = calc_return(market_data.get('sh_index'), prev_date, end_date)
        result['hs300_return'] = calc_return(market_data.get('hs300'), prev_date, end_date)
        result['zz500_return'] = calc_return(market_data.get('zz500'), prev_date, end_date)
        result['zz1000_return'] = calc_return(market_data.get('zz1000'), prev_date, end_date)
        result['cyb_return'] = calc_return(market_data.get('cyb'), prev_date, end_date)

        # 2. 期末指数点位
        result['sh_index_close'] = market_data.get('sh_index', {}).get(end_date, {}).get('close')
        result['hs300_close'] = market_data.get('hs300', {}).get(end_date)

        # 3. 期货涨跌幅
        result['if_return'] = calc_return(market_data.get('if2606'), prev_date, end_date)
        result['ic_return'] = calc_return(market_data.get('ic2606'), prev_date, end_date)
        result['im_return'] = calc_return(market_data.get('im2606'), prev_date, end_date)

        # 4. 期货期末价格（用于计算基差）
        result['if_close'] = market_data.get('if2606', {}).get(end_date)
        result['if_far_close'] = market_data.get('if2612', {}).get(end_date)  # 最远月合约

        # 5. 计算当月合约基差（IF升贴水）
        if result['hs300_close'] and result['if_close']:
            result['basis'] = result['hs300_close'] - result['if_close']
            result['basis_pct'] = result['basis'] / result['hs300_close'] if result['hs300_close'] else 0
        else:
            result['basis'] = None
            result['basis_pct'] = None

        # 6. 计算最远月合约基差
        if result['hs300_close'] and result['if_far_close']:
            result['basis_far'] = result['hs300_close'] - result['if_far_close']
            result['basis_far_pct'] = result['basis_far'] / result['hs300_close'] if result['hs300_close'] else 0
        else:
            result['basis_far'] = None
            result['basis_far_pct'] = None

        # 7. 计算基差变化（期初 vs 期末）
        hs300_start = market_data.get('hs300', {}).get(prev_date)
        if2606_start = market_data.get('if2606', {}).get(prev_date)
        if2612_start = market_data.get('if2612', {}).get(prev_date)

        # 当月合约基差变化
        if hs300_start and if2606_start and result['hs300_close'] and result['if_close']:
            basis_start = hs300_start - if2606_start
            basis_end = result['hs300_close'] - result['if_close']
            result['basis_change'] = basis_end - basis_start
        else:
            result['basis_change'] = None

        # 最远月合约基差变化
        if hs300_start and if2612_start and result['hs300_close'] and result['if_far_close']:
            basis_far_start = hs300_start - if2612_start
            basis_far_end = result['hs300_close'] - result['if_far_close']
            result['basis_far_change'] = basis_far_end - basis_far_start
        else:
            result['basis_far_change'] = None

        # 8. 日均成交额（万亿）
        sh_index_data = market_data.get('sh_index', {})
        turnovers = []
        for d in dates:
            if d in sh_index_data and 'turnover' in sh_index_data[d]:
                turnovers.append(sh_index_data[d]['turnover'])
        if turnovers:
            result['avg_turnover'] = sum(turnovers) / len(turnovers) / 1e12  # 转为万亿
        else:
            result['avg_turnover'] = None

        # 7. 产品贡献汇总（从module3_data求和）
        total_stk_contrib = 0
        total_fut_contrib = 0
        for m3 in module3_data:
            if m3.get('stk_contrib') is not None:
                total_stk_contrib += m3['stk_contrib']
            if m3.get('fut_contrib') is not None:
                total_fut_contrib += m3['fut_contrib']
        result['total_stk_contrib'] = total_stk_contrib
        result['total_fut_contrib'] = total_fut_contrib

        return result
