"""
主运行脚本 - 生成量化基金产品报告

本脚本是报告生成的入口，执行流程：
1. 读取本地数据（净值、期货、持仓等）
2. 获取米筐数据（股票价格、因子数据）
3. 计算报告指标
4. 保存中间数据
5. 生成PDF报告

所有配置参数均从config.py读取，便于维护和修改。
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_reader import DataReader
from rq_fetcher import RQDataFetcher
from calculator import ReportCalculator
from pdf_generator import PDFGenerator


def _fmt_pct(val):
    """格式化百分比值，处理 None"""
    if val is None:
        return 'N/A'
    return f'{val * 100:.2f}%'


_logger = logging.getLogger(__name__)


def _convert_stock_code(code):
    """
    转换股票代码格式

    根据配置中的股票代码规则进行转换：
    - 60开头 → XSHG（沪市）
    - 688开头 → XSHG（科创板）
    - 其他 → XSHE（深市）

    参数:
        code: 股票代码（可以是整数、字符串，带或不带后缀）

    返回:
        str: 完整的股票代码, 如 '600000.XSHG'
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


def save_intermediate_data(calc_result, module2_data, module3_data, module4_data, module5_data, module6_data, module7_data, daily_data, output_dir):
    """
    保存中间数据到文件，方便核对
    
    参数:
        calc_result: 模块1计算结果（净值信息）
        module2_data: 模块2数据（每日收益明细）
        module3_data: 模块3数据（多空详情）
        module4_data: 模块4数据（Top10权重）
        module5_data: 模块5数据（持仓集中度）
        module6_data: 模块6数据（因子敞口）
        daily_data: 每日基础数据
        output_dir: 输出目录路径
    """
    data_dir = os.path.join(output_dir, "intermediate_data")
    os.makedirs(data_dir, exist_ok=True)

    # 模块1：净值曲线
    pd.DataFrame([calc_result['nav_curve']]).T.to_csv(
        os.path.join(data_dir, "nav_curve.csv"), encoding='utf-8-sig'
    )

    # 基准曲线
    pd.DataFrame([calc_result['bench_curve']]).T.to_csv(
        os.path.join(data_dir, "bench_curve.csv"), encoding='utf-8-sig'
    )

    # 模块1汇总
    pd.DataFrame([calc_result['nav_curve']]).T.to_csv(
        os.path.join(data_dir, "module1_summary.csv"), encoding='utf-8-sig'
    )

    # 模块2：每日收益
    pd.DataFrame(module2_data).to_csv(
        os.path.join(data_dir, "module2_daily_return.csv"), encoding='utf-8-sig', index=False
    )

    # 模块3：多空详情
    module3_flat = []
    for m3 in module3_data:
        row = {
            'date': m3['date'],
            'total_stk_return': m3['total_stk_return'],
            'fut_return': m3['fut_return'],
            'basis_return': m3['basis_return'],
            'stk_contrib': m3['stk_contrib'],
            'fut_contrib': m3['fut_contrib'],
            'total_contrib': m3['total_contrib'],
            'product_assets': m3['product_assets'],
        }
        for i, acc in enumerate(m3['accounts']):
            row[f'acc{i}_name'] = acc['account']
            row[f'acc{i}_return'] = acc['return']
            row[f'acc{i}_excess'] = acc['excess']
            row[f'acc{i}_pnl'] = acc['pnl']
            row[f'acc{i}_net_assets'] = acc['net_assets']
        module3_flat.append(row)
    pd.DataFrame(module3_flat).to_csv(
        os.path.join(data_dir, "module3_detail.csv"), encoding='utf-8-sig', index=False
    )

    # 每日净值数据
    pd.DataFrame([daily_data['nav']]).T.to_csv(
        os.path.join(data_dir, "nav_by_date.csv"), encoding='utf-8-sig'
    )

    # 每日期货数据
    pd.DataFrame([daily_data['fut']]).T.to_csv(
        os.path.join(data_dir, "fut_by_date.csv"), encoding='utf-8-sig'
    )

    # 股票账户详情
    stk_df_rows = []
    for date, accounts in daily_data['stk_accounts'].items():
        for acc, info in accounts.items():
            if info is not None:
                row = info.to_dict()
                row['date'] = date
                row['account'] = acc
                stk_df_rows.append(row)
    if stk_df_rows:
        pd.DataFrame(stk_df_rows).to_csv(
            os.path.join(data_dir, "stk_accounts_detail.csv"), encoding='utf-8-sig', index=False
        )

    # 模块4：Top10权重
    module4_flat = []
    for date_str, data in module4_data.items():
        for i, item in enumerate(data.get('top10', [])):
            row = {
                'date': date_str,
                'rank': i + 1,
                'code': item.get('code'),
                'hold': item.get('hold'),
                'market_value': item.get('market_value'),
                'weight': item.get('weight'),
            }
            module4_flat.append(row)
    if module4_flat:
        pd.DataFrame(module4_flat).to_csv(
            os.path.join(data_dir, "module4_top10.csv"), encoding='utf-8-sig', index=False
        )

    # 模块5：持仓集中度
    module5_flat = []
    for date_str, data in module5_data.items():
        row = {
            'date': date_str,
            'total_stocks': data['total_stocks'],
            'top_20_pct_count': data['top_20_pct_count'],
            'top_20_pct_weight': data['top_20_pct_weight'],
        }
        module5_flat.append(row)
    if module5_flat:
        pd.DataFrame(module5_flat).to_csv(
            os.path.join(data_dir, "module5_concentration.csv"), encoding='utf-8-sig', index=False
        )

    # 模块6：因子数据
    with open(os.path.join(data_dir, "module6_factors.json"), 'w', encoding='utf-8') as f:
        json.dump(module6_data, f, ensure_ascii=False, indent=2)

    # 模块7：行业对比数据
    module7_flat = []
    for date_str, data in module7_data.items():
        for industry, weights in data.get('portfolio_industry', {}).items():
            module7_flat.append({
                'date': date_str,
                'industry': industry,
                'portfolio_weight': weights,
                'benchmark_weight': data.get('benchmark_industry', {}).get(industry, 0),
                'diff': weights - data.get('benchmark_industry', {}).get(industry, 0),
            })
    if module7_flat:
        pd.DataFrame(module7_flat).to_csv(
            os.path.join(data_dir, "module7_industry.csv"), encoding='utf-8-sig', index=False
        )

    _logger.info("中间数据已保存到: %s", data_dir)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    """
    主函数 - 执行报告生成的完整流程
    
    执行步骤：
    1. 连接米筐获取前一交易日
    2. 读取本地数据
    3. 获取米筐数据（股票价格、因子数据、行业分类）
    4. 计算报告指标（含行业对比模块7）
    5. 保存中间数据
    6. 生成PDF报告
    """
    _logger.info("=" * 60)
    _logger.info("量化基金产品周期报告生成系统")
    _logger.info("=" * 60)

    _logger.info("产品: %s (%s)", config.REPORT_PRODUCT_NAME, config.REPORT_PRODUCT_CODE)
    _logger.info("周期: %s ~ %s", config.REPORT_START, config.REPORT_END)
    _logger.info("基准: %s (%s)", config.BENCHMARK_CODE, config.BENCHMARK_NAME)

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # [1/7] 先连接米筐获取前一交易日
    _logger.info("步骤 [1/7] 连接米筐获取前一交易日")
    rq_fetcher = RQDataFetcher()
    prev_date = None

    if rq_fetcher.connect():
        try:
            prev_date = rq_fetcher.get_previous_trading_date(config.REPORT_START)
            _logger.info("前一交易日: %s", prev_date)
        except Exception as e:
            _logger.error("获取前一交易日失败: %s", e)
    
    if not prev_date:
        # 回退：简单跳过周末
        from datetime import timedelta
        start_dt = datetime.strptime(config.REPORT_START, "%Y%m%d")
        prev_dt = start_dt - timedelta(days=1)
        while prev_dt.weekday() >= 5:
            prev_dt -= timedelta(days=1)
        prev_date = prev_dt.strftime("%Y%m%d")
        _logger.warning("使用兜底方式获取前一交易日: %s", prev_date)

    # [2/7] 读取本地数据
    _logger.info("步骤 [2/7] 读取本地数据")
    reader = DataReader(prev_date=prev_date)
    # 用米筐真实交易日历覆盖本地简单跳过周末的计算
    if rq_fetcher.connected:
        try:
            from rq_fetcher import normalize_date_for_rq
            start_rq = normalize_date_for_rq(config.REPORT_START)
            end_rq = normalize_date_for_rq(config.REPORT_END)
            real_dates = rq_fetcher.client.get_trading_dates(start_rq, end_rq)
            if real_dates is not None and len(real_dates) > 0:
                real_date_strs = [d.strftime('%Y%m%d') for d in real_dates]
                reader.set_trading_calendar(real_date_strs)
                _logger.info("使用米筐交易日历: %d个交易日", len(real_date_strs))
        except Exception as e:
            _logger.warning("获取真实交易日历失败，使用本地计算: %s", e)


    daily_data = reader.collect_daily_data()
    positions_data = reader.collect_positions_data()

    _logger.info("交易日期: %s", daily_data['dates'])
    _logger.info("账户列表: %s", daily_data['accounts'])
    _logger.info("持仓数据天数: %d", len(positions_data))

    # 包含前一交易日用于计算收益率
    dates_with_prev = [prev_date] + daily_data['dates']

    # [3/7] 获取米筐数据
    _logger.info("步骤 [3/7] 获取米筐数据")

    if not rq_fetcher.connected:
        _logger.warning("米筐未连接，使用本地数据计算...")
        benchmark_prices = {}
        stock_prices = {}
        style_factors = {}
        benchmark_factors = {}
        industry_data = {}
    else:
        # 获取基准价格
        benchmark_prices = rq_fetcher.get_benchmark_prices(dates_with_prev)
        _logger.info("基准价格: %d天", len(benchmark_prices) if benchmark_prices else 0)

        # 获取所有股票代码
        all_stock_codes = set()
        for d, pos_df in positions_data.items():
            if pos_df is not None:
                for code in pos_df['code'].tolist():
                    full_code = _convert_stock_code(code)
                    all_stock_codes.add(full_code)

        stock_codes_list = list(all_stock_codes)

        if stock_codes_list:
            stock_prices = rq_fetcher.get_stock_prices(stock_codes_list, dates_with_prev)
            _logger.info("股票价格: %d只", len(stock_prices) if stock_prices else 0)

            # 获取因子数据
            style_factors = {}
            benchmark_factors = {}
            for d in daily_data['dates']:
                codes = []
                if d in positions_data and positions_data[d] is not None:
                    codes = [_convert_stock_code(c) for c in positions_data[d]['code'].tolist()[:50]]

                if codes:
                    sf = rq_fetcher.get_style_factor_exposure(codes, d)
                    if sf:
                        style_factors[d] = sf

                bf = rq_fetcher.get_benchmark_factor_exposure(d)
                if bf:
                    benchmark_factors[d] = bf

            _logger.info("因子数据: 组合%d天, 基准%d天", len(style_factors), len(benchmark_factors))

            # 获取行业分类数据
            industry_data = {}
            for d in daily_data['dates']:
                codes = []
                if d in positions_data and positions_data[d] is not None:
                    codes = [_convert_stock_code(c) for c in positions_data[d]['code'].tolist()]

                if codes:
                    ind = rq_fetcher.get_industry_classification(codes, d)
                    if ind:
                        industry_data[d] = ind

                bench_ind = rq_fetcher.get_benchmark_industry_weights(d)
                if bench_ind:
                    industry_data[f'{d}_bench'] = bench_ind

            _logger.info("行业分类: 组合%d天, 基准%d天", 
                         sum(1 for k in industry_data if "_bench" not in k),
                         sum(1 for k in industry_data if "_bench" in k))

            # 获取市场宏观数据（用于市场回顾文字）
            _logger.info("获取市场宏观数据...")
            market_overview_data = rq_fetcher.get_market_overview_data(dates_with_prev)
            if market_overview_data:
                _logger.info("  上证指数: %d天", len(market_overview_data.get('sh_index', {})))
                _logger.info("  沪深300: %d天", len(market_overview_data.get('hs300', {})))
                fut_items = [(k, v) for k, v in market_overview_data.items() if k.startswith(('if', 'ic', 'im'))]
                if fut_items:
                    _logger.info("  期货: %s", ' '.join([f'{k.upper()}:{len(v)}' for k, v in fut_items]))
            else:
                market_overview_data = {}
        else:
            stock_prices = {}
            style_factors = {}
            benchmark_factors = {}
            industry_data = {}
            market_overview_data = {}

    # [4/7] 计算报告数据
    _logger.info("步骤 [4/7] 计算报告数据")
    calculator = ReportCalculator(
        daily_data=daily_data,
        benchmark_prices=benchmark_prices,
        stock_prices=stock_prices,
        style_factors=style_factors,
        benchmark_factors=benchmark_factors
    )

    calc_result = calculator.calc_module1()
    _logger.info("模块1: 净值增长=%s, 基准=%s",
                 _fmt_pct(calc_result.get('nav_growth')),
                 _fmt_pct(calc_result.get('bench_growth')))

    module2_data = calculator.calc_module2()
    _logger.info("模块2: %d天收益率数据", len(module2_data))

    module3_data = calculator.calc_module3()
    _logger.info("模块3: %d天多空详情", len(module3_data))

    module4_data = calculator.calc_module4(positions_data, stock_prices)
    _logger.info("模块4: %d天Top10数据", len(module4_data))

    module5_data = calculator.calc_module5(positions_data, stock_prices)
    _logger.info("模块5: %d天集中度数据", len(module5_data))

    module7_data = calculator.calc_module7(positions_data, stock_prices, industry_data)
    _logger.info("模块7: %d天行业对比数据", len(module7_data))

    _logger.info("模块6: 从本地parquet读取因子数据...")
    factor_exposure_data = {}
    for d in daily_data['dates']:
        # 获取组合因子暴露（持仓加权）
        if d in positions_data and positions_data[d] is not None:
            portfolio_factors = reader.calculate_portfolio_factor_exposure(d, positions_data[d])
        else:
            portfolio_factors = None
        
        # 获取基准因子暴露
        benchmark_factors_local = reader.get_benchmark_factor_exposure(d)
        
        if portfolio_factors or benchmark_factors_local:
            factor_exposure_data[d] = {
                'portfolio': portfolio_factors or {},
                'benchmark': benchmark_factors_local or {},
            }
    
    if factor_exposure_data:
        _logger.info("  本地因子数据: %d天", len(factor_exposure_data))
        _logger.info("  因子列表: %s", list(config.FactorConfig.SELECTED_FACTORS.keys()))
    
    module6_data = calculator.calc_module6(positions_data, stock_prices, factor_exposure_data)
    _logger.info("模块6: %d天因子数据", len(module6_data))

    market_review_data = calculator.calc_market_review_data(market_overview_data, module3_data)
    if market_review_data:
        _logger.info("市场回顾: 上证%.2f%%, 沪深300%.2f%%", 
                    market_review_data.get('sh_index_return', 0)*100,
                    market_review_data.get('hs300_return', 0)*100)
        _logger.info("         股票超额%.2f%%, 基差贡献%.2f%%",
                    market_review_data.get('total_stk_contrib', 0)*100,
                    market_review_data.get('total_fut_contrib', 0)*100)

    # [5/7] 保存中间数据
    _logger.info("步骤 [5/7] 保存中间数据")
    save_intermediate_data(
        calc_result, module2_data, module3_data,
        module4_data, module5_data, module6_data, module7_data,
        daily_data, config.OUTPUT_DIR
    )

    # [6/7] 生成PDF报告
    _logger.info("步骤 [6/7] 生成PDF报告")

    # 生成输出文件名
    output_filename = config.get_report_filename(
        config.REPORT_PRODUCT_CODE, 
        config.REPORT_START, 
        config.REPORT_END
    )
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)

    # 生成PDF报告（模块顺序：1,2,4,5,7,6）
    pdf_gen = PDFGenerator(output_path)
    pdf_gen.build(
        calc_result=calc_result,
        module2_data=module2_data,
        module3_data=module3_data,
        module4_data=module4_data,
        module5_data=module5_data,
        module6_data=module6_data,
        module7_data=module7_data,
        dates=daily_data['dates'],
        market_review_data=market_review_data
    )

    _logger.info("=" * 60)
    _logger.info("报告生成完成! 输出路径: %s", output_path)
    _logger.info("=" * 60)


if __name__ == "__main__":
    main()