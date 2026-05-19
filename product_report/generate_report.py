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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_reader import DataReader
from rq_fetcher import RQDataFetcher
from calculator import ReportCalculator
from pdf_generator import PDFGenerator


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


def save_intermediate_data(calc_result, module2_data, module3_data, module4_data, module5_data, module6_data, daily_data, output_dir):
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

    print(f"  中间数据已保存到: {data_dir}")


def main():
    """
    主函数 - 执行报告生成的完整流程
    
    执行步骤：
    1. 初始化配置
    2. 读取本地数据
    3. 获取米筐数据
    4. 计算报告指标
    5. 保存中间数据
    6. 生成PDF报告
    """
    print("=" * 60)
    print("量化基金产品报告生成器")
    print("=" * 60)

    print(f"\n产品: {config.REPORT_PRODUCT_NAME} ({config.REPORT_PRODUCT_CODE})")
    print(f"周期: {config.REPORT_START} ~ {config.REPORT_END}")
    print(f"基准: {config.BENCHMARK_CODE} ({config.BENCHMARK_NAME})")

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/6] 读取本地数据...")
    reader = DataReader()

    daily_data = reader.collect_daily_data()
    positions_data = reader.collect_positions_data()

    print(f"  交易日期: {daily_data['dates']}")
    print(f"  账户列表: {daily_data['accounts']}")
    print(f"  持仓数据天数: {len(positions_data)}")

    # 包含前一交易日用于计算收益率
    dates_with_prev = [config.REPORT_PREV] + daily_data['dates']

    print("\n[2/6] 获取米筐数据...")

    rq_fetcher = RQDataFetcher()

    if not rq_fetcher.connect():
        print("  米筐连接失败，使用模拟数据继续...")
        benchmark_prices = {}
        stock_prices = {}
        style_factors = {}
        benchmark_factors = {}
    else:
        # 获取基准价格
        benchmark_prices = rq_fetcher.get_benchmark_prices(dates_with_prev)
        print(f"  基准价格获取: {len(benchmark_prices) if benchmark_prices else 0}条")

        # 获取所有股票代码
        all_stock_codes = set()
        for d, pos_df in positions_data.items():
            if pos_df is not None:
                for code in pos_df['code'].tolist():
                    full_code = _convert_stock_code(code)
                    all_stock_codes.add(full_code)

        stock_codes_list = list(all_stock_codes)

        if stock_codes_list:
            # 获取股票价格
            stock_prices = rq_fetcher.get_stock_prices(stock_codes_list, dates_with_prev)
            print(f"  股票价格获取: {len(stock_prices) if stock_prices else 0}只股票")

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

            print(f"  因子数据: 组合{len(style_factors)}天, 基准{len(benchmark_factors)}天")
        else:
            stock_prices = {}
            style_factors = {}
            benchmark_factors = {}

    print("\n[3/6] 计算报告数据...")

    calculator = ReportCalculator(
        daily_data=daily_data,
        benchmark_prices=benchmark_prices,
        stock_prices=stock_prices,
        style_factors=style_factors,
        benchmark_factors=benchmark_factors
    )

    # 计算各模块
    calc_result = calculator.calc_module1()
    print(f"  模块1: 净值增长={calc_result['nav_growth']:.4f}, 基准={calc_result['bench_growth']:.4f}")

    module2_data = calculator.calc_module2()
    print(f"  模块2: {len(module2_data)}天收益率数据")

    module3_data = calculator.calc_module3()
    print(f"  模块3: {len(module3_data)}天多空详情")

    module4_data = calculator.calc_module4(positions_data, stock_prices)
    print(f"  模块4: {len(module4_data)}天Top10数据")

    module5_data = calculator.calc_module5(positions_data, stock_prices)
    print(f"  模块5: {len(module5_data)}天集中度数据")

    module6_data = calculator.calc_module6(positions_data, stock_prices)
    print(f"  模块6: {len(module6_data)}天因子数据")

    print("\n[4/6] 保存中间数据...")
    save_intermediate_data(
        calc_result, module2_data, module3_data,
        module4_data, module5_data, module6_data,
        daily_data, config.OUTPUT_DIR
    )

    print("\n[5/6] 生成PDF报告...")

    # 生成输出文件名
    output_filename = config.get_report_filename(
        config.REPORT_PRODUCT_CODE, 
        config.REPORT_START, 
        config.REPORT_END
    )
    output_path = os.path.join(config.OUTPUT_DIR, output_filename)

    # 生成PDF报告
    pdf_gen = PDFGenerator(output_path)
    pdf_gen.build(
        calc_result=calc_result,
        module2_data=module2_data,
        module3_data=module3_data,
        module4_data=module4_data,
        module5_data=module5_data,
        module6_data=module6_data,
        dates=daily_data['dates']
    )

    print("\n" + "=" * 60)
    print("报告生成完成!")
    print(f"输出路径: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()