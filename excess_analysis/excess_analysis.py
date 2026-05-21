"""
超额收益分析工具
================
分析实际持仓和理论持仓的超额收益，输出数据表和图表。

用法:
    python excess_analysis.py --start 20250912 --end 20260519
    python excess_analysis.py --freq daily               # 默认：日度
    python excess_analysis.py --freq weekly              # 周度聚合模式
    python excess_analysis.py --start 20260101           # 从指定日期到最新

输出:
    - data/*_excess.csv           实际/理论持仓时序数据
    - charts/*_excess.png         实际/理论持仓超额曲线图
    （周度模式文件名含 _weekly 后缀）
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import warnings

from pathlib import Path
# 自动把 D:\code 加入 Python 搜索路径
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')
from ultis.email_manager import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗口
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ============================================================
# 配置区（可根据需要修改）
# ============================================================
# 项目根目录
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    'actual_dir': r'\\192.168.3.100\samba\data\out\stk_fut',
    'target_dir': r'\\192.168.3.100\samba\data\target',
    'output_dir': _PROJECT_DIR,
    'rq_username': 'license',
    'rq_password': 'jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=',
    'benchmark_code': '000300.XSHG',  # 沪深300
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='超额收益分析工具 —— 实际持仓 vs 理论持仓'
    )
    parser.add_argument('--start', type=str, default=None,
                        help='开始日期 (YYYYMMDD)，不指定则从最早共有日期开始')
    parser.add_argument('--end', type=str, default=None,
                        help='结束日期 (YYYYMMDD)，不指定则到最晚共有日期结束')
    parser.add_argument('--freq', type=str, default='daily',
                        choices=['daily', 'weekly'],
                        help='数据频率: daily(日度) / weekly(周度) (默认: daily)')
    parser.add_argument('--output', type=str, default=CONFIG['output_dir'],
                        help=f'输出目录 (默认: {CONFIG["output_dir"]})')
    return parser.parse_args()


def get_common_trading_dates(start_date=None, end_date=None):
    """
    获取实际持仓和理论持仓共有的日期列表，按日期范围过滤。
    返回 (dates_list, actual_all_dates, target_all_dates)
    """
    actual_dir = CONFIG['actual_dir']
    target_dir = CONFIG['target_dir']

    # 实际持仓：找包含 haitong + PBHSZX1H 的日期
    actual_dates = set()
    for d in sorted(os.listdir(actual_dir)):
        dpath = os.path.join(actual_dir, d)
        if not os.path.isdir(dpath):
            continue
        files = os.listdir(dpath)
        if any('haitong' in f.lower() and 'PBHSZX1H' in f for f in files):
            actual_dates.add(d)

    # 理论持仓：找包含 TCHMD 的日期
    target_dates = set()
    for d in sorted(os.listdir(target_dir)):
        dpath = os.path.join(target_dir, d)
        if not os.path.isdir(dpath):
            continue
        files = os.listdir(dpath)
        if any('TCHMD' in f for f in files):
            target_dates.add(d)

    common = sorted(actual_dates & target_dates)
    print(f"实际持仓有效交易日: {len(actual_dates)} ({min(actual_dates)} ~ {max(actual_dates)})")
    print(f"理论持仓有效交易日: {len(target_dates)} ({min(target_dates)} ~ {max(target_dates)})")
    print(f"共有交易日: {len(common)} ({common[0]} ~ {common[-1]})")

    # 按起止日期过滤
    if start_date:
        common = [d for d in common if d >= start_date]
    if end_date:
        common = [d for d in common if d <= end_date]

    if common:
        print(f"过滤后交易日: {len(common)} ({common[0]} ~ {common[-1]})")
    else:
        print("过滤后交易日: 0 (无匹配日期)")
    return common, sorted(actual_dates), sorted(target_dates)


def load_actual_data(dates):
    """
    加载实际持仓数据。
    返回 DataFrame: [date, long_return, bench_return, excess]
    """
    records = []
    actual_dir = CONFIG['actual_dir']

    for d in dates:
        dpath = os.path.join(actual_dir, d)
        files = [f for f in os.listdir(dpath)
                 if 'haitong' in f.lower() and 'PBHSZX1H' in f]
        if not files:
            continue
        fpath = os.path.join(dpath, files[0])
        try:
            df = pd.read_csv(fpath, encoding='utf-8', header=0)
            row = df.iloc[0]
            # 列21 = 多头当日收益率, 列24 = 基准收益率
            long_ret = row.iloc[21]   # 多头收益率
            bench_ret = row.iloc[24]  # 基准收益率
            excess = long_ret - bench_ret
            records.append({
                'date': d,
                'long_return': float(long_ret),
                'bench_return': float(bench_ret),
                'excess': float(excess),
            })
        except Exception as e:
            print(f"  [WARN] 读取 {d} 失败: {e}")

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result['date'] = pd.to_datetime(result['date'], format='%Y%m%d')
    result = result.sort_values('date').reset_index(drop=True)
    print(f"实际持仓数据加载完成: {len(result)} 条")
    return result


def find_target_file(d, target_dir):
    """查找日期 d 对应的 TCHMD 文件"""
    dpath = os.path.join(target_dir, d)
    if not os.path.isdir(dpath):
        return None
    files = [f for f in os.listdir(dpath) if 'TCHMD' in f]
    if not files:
        return None
    # 如果有多个 TCHMD 文件，优先选 zz1800，其次 000906
    zz = [f for f in files if 'zz1800' in f]
    if zz:
        return os.path.join(dpath, zz[0])
    return os.path.join(dpath, files[0])


def load_theoretical_data(dates):
    """
    加载理论持仓数据，从米筐获取行情计算收益。
    返回 DataFrame: [date, portfolio_return, bench_return, excess]

    优化策略：
    1. 收集所有日期内所有出现的股票代码
    2. 一次性从米筐获取全部代码在全部日期区间的行情
    3. 逐日计算组合收益
    """
    import rqdatac as rq

    target_dir = CONFIG['target_dir']
    benchmark_code = CONFIG['benchmark_code']

    # 初始化米筐
    rq.init(CONFIG['rq_username'], CONFIG['rq_password'])
    print("米筐 rqdatac 初始化成功")

    # ---------- 第一步：读取所有持仓权重 ----------
    print("\n读取理论持仓文件...")
    all_weights = {}  # date -> DataFrame(code, w)
    all_codes = set()

    for d in dates:
        fpath = find_target_file(d, target_dir)
        if fpath is None:
            print(f"  [WARN] {d} 无 TCHMD 文件")
            continue
        try:
            df = pd.read_csv(fpath, encoding='gbk')
            # 检查必要列
            if 'code' not in df.columns or 'w' not in df.columns:
                print(f"  [WARN] {d} 文件列名异常: {list(df.columns)}")
                continue
            wdf = df[['code', 'w']].copy()
            wdf['w'] = wdf['w'].astype(float)
            # 权重归一化（防止四舍五入导致和不等于1）
            w_sum = wdf['w'].sum()
            if abs(w_sum - 1.0) > 1e-6 and w_sum > 0:
                wdf['w'] = wdf['w'] / w_sum
            all_weights[d] = wdf
            all_codes.update(wdf['code'].tolist())
        except Exception as e:
            print(f"  [WARN] 读取 {d} 权重失败: {e}")

    print(f"  共 {len(all_weights)} 个交易日，{len(all_codes)} 只唯一个股")

    if not all_weights:
        print("[ERROR] 没有加载到任何理论持仓数据")
        return pd.DataFrame()

    # ---------- 第二步：批量获取个股行情 ----------
    codes_list = sorted(all_codes)
    start_dt = dates[0]
    end_dt = dates[-1]

    # 转换成 rqdatac 可读的日期格式（往前多取几天保证有前一日收盘价）
    rq_start = (datetime.strptime(start_dt, '%Y%m%d') - timedelta(days=10)).strftime('%Y-%m-%d')
    rq_end = datetime.strptime(end_dt, '%Y%m%d').strftime('%Y-%m-%d')

    print(f"\n从米筐批量获取行情: {len(codes_list)} 只股票, {rq_start} ~ {rq_end}")
    print("  获取收盘价...", end=' ', flush=True)

    # 批量获取所有股票收盘价
    try:
        price_df = rq.get_price(
            codes_list,
            start_date=rq_start,
            end_date=rq_end,
            fields=['close'],
            expect_df=True,
        )
        print(f"完成 ({price_df.shape[0] if hasattr(price_df, 'shape') else 'ok'} 行)")
    except Exception as e:
        print(f"\n  [WARN] 批量获取行情失败: {e}")
        print("  尝试分批获取...")
        price_dfs = []
        batch_size = 50
        for i in range(0, len(codes_list), batch_size):
            batch = codes_list[i:i+batch_size]
            try:
                pdf = rq.get_price(batch, start_date=rq_start, end_date=rq_end, fields=['close'], expect_df=True)
                price_dfs.append(pdf)
                print(f"    batch {i//batch_size+1}/{(len(codes_list)-1)//batch_size+1} OK ({pdf.shape[0]} rows)")
            except Exception as e2:
                print(f"    batch {i//batch_size+1} FAILED: {e2}")
        if price_dfs:
            price_df = pd.concat(price_dfs)
        else:
            print("[ERROR] 无法获取行情数据")
            return pd.DataFrame()

    # 行情数据格式处理：MultiIndex (order_book_id, date) -> 扁平化
    if isinstance(price_df.index, pd.MultiIndex):
        price_df = price_df.reset_index()
        # 列名通常是 index level names
        col_names = list(price_df.columns)
        # 寻找包含 code/date/close 的列
        close_col = None
        for c in col_names:
            if c is not None and 'close' in str(c).lower():
                close_col = c
                break
        code_col = col_names[0]
        date_col = col_names[1]
        price_df = price_df.rename(columns={
            code_col: 'code',
            date_col: 'date',
            close_col: 'close' if close_col else col_names[2],
        })
        price_df = price_df[['code', 'date', 'close']]
    elif 'close' not in price_df.columns:
        # 尝试找 close 列
        for c in price_df.columns:
            if 'close' in str(c).lower():
                price_df = price_df.rename(columns={c: 'close'})
                break

    price_df['date'] = pd.to_datetime(price_df['date'])
    print(f"  行情数据形状: {price_df.shape}")

    # 获取基准指数行情
    print("  获取基准指数(沪深300)行情...", end=' ', flush=True)
    try:
        bench_raw = rq.get_price(
            benchmark_code,
            start_date=rq_start,
            end_date=rq_end,
            fields=['close'],
        )
        if isinstance(bench_raw.index, pd.MultiIndex):
            bench_df = bench_raw.reset_index()
            bench_df.columns = ['code', 'date', 'close']
        else:
            bench_df = bench_raw.reset_index()
            bench_df.columns = ['date', 'close']
            bench_df['code'] = benchmark_code
        bench_df['date'] = pd.to_datetime(bench_df['date'])
        print(f"完成 ({bench_df.shape[0]} 行)")
    except Exception as e:
        print(f"失败: {e}")
        bench_df = None

    # ---------- 第三步：按日期计算组合收益 ----------
    print("\n计算每日组合收益率...")
    results = []

    for i, d in enumerate(dates):
        if d not in all_weights:
            continue

        wdf = all_weights[d]
        current_date = pd.Timestamp(datetime.strptime(d, '%Y%m%d'))

        # 获取当日行情
        day_prices = price_df[price_df['date'] == current_date]
        if day_prices.empty:
            print(f"  [WARN] {d} 无行情数据，跳过")
            continue

        # 合并权重和行情
        merged = wdf.merge(day_prices[['code', 'close']], on='code', how='inner')
        if len(merged) < len(wdf) * 0.5:
            missing = set(wdf['code']) - set(merged['code'])
            print(f"  [WARN] {d} 行情匹配率 {len(merged)}/{len(wdf)}，缺失 {len(missing)} 只")
            if missing:
                print(f"         缺失示例: {list(missing)[:5]}")

        # 找上一个有行情数据的交易日
        prev_date = None
        for j in range(i - 1, -1, -1):
            prev_check = price_df[price_df['date'] == pd.Timestamp(datetime.strptime(dates[j], '%Y%m%d'))]
            if not prev_check.empty:
                prev_date = pd.Timestamp(datetime.strptime(dates[j], '%Y%m%d'))
                break

        if prev_date is None:
            print(f"  [WARN] {d} 找不到前一交易日行情")
            continue

        prev_prices = price_df[price_df['date'] == prev_date]

        # 计算个股收益率
        merged = merged.merge(prev_prices[['code', 'close']], on='code',
                              how='inner', suffixes=('', '_prev'))
        if merged.empty:
            print(f"  [WARN] {d} 前一交易日无匹配数据")
            continue

        merged['stock_return'] = merged['close'] / merged['close_prev'] - 1

        # 组合收益率 = sum(收益率 * 权重)
        portfolio_ret = (merged['stock_return'] * merged['w']).sum()

        # 基准指数收益率
        if bench_df is not None:
            bench_today = bench_df[bench_df['date'] == current_date]
            bench_prev = bench_df[bench_df['date'] == prev_date]
            if not bench_today.empty and not bench_prev.empty:
                bench_ret = (bench_today['close'].values[0]
                             / bench_prev['close'].values[0] - 1)
            else:
                bench_ret = np.nan
        else:
            bench_ret = np.nan

        excess = portfolio_ret - bench_ret if not np.isnan(bench_ret) else np.nan

        results.append({
            'date': d,
            'portfolio_return': portfolio_ret,
            'bench_return': bench_ret,
            'excess': excess,
        })

        if (i + 1) % 30 == 0:
            print(f"  ... 已处理 {i+1}/{len(dates)} 天")

    result = pd.DataFrame(results)
    if result.empty:
        return result
    result['date'] = pd.to_datetime(result['date'], format='%Y%m%d')
    result = result.sort_values('date').reset_index(drop=True)
    print(f"理论持仓数据计算完成: {len(result)} 条")
    return result


def set_chinese_font():
    """
    设置中文字体。
    直接扫描 fontManager 中的字体列表匹配已知中文字体名，
    避免 findfont(fallback_to_default=False) 在不匹配时抛异常的问题。
    """
    import matplotlib.font_manager as fm

    # 优先搜索的字体名（按偏好顺序）
    preferred = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'STXihei',
                 'FangSong', 'KaiTi', 'STSong']

    # 扫描已注册字体
    registered_names = {f.name for f in fm.fontManager.ttflist}

    for name in preferred:
        if name in registered_names:
            # 将找到的字体设为第1优先级
            sans_list = list(plt.rcParams['font.sans-serif'])
            if name in sans_list:
                sans_list.remove(name)
            plt.rcParams['font.sans-serif'] = [name] + sans_list
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  中文字体: {name}")
            return name

    # Fallback: 通过字体文件名直接加载
    for target in ['simhei.ttf', 'msyh.ttc', 'NotoSansSC-VF.ttf', 'simfang.ttf']:
        for f in fm.fontManager.ttflist:
            if target in f.fname.lower():
                plt.rcParams['font.sans-serif'] = [f.name] + list(plt.rcParams['font.sans-serif'])
                plt.rcParams['axes.unicode_minus'] = False
                print(f"  中文字体(fallback): {f.name} ({f.fname})")
                return f.name

    print("  [WARN] 未找到中文字体，图表中文可能显示为方框")
    plt.rcParams['axes.unicode_minus'] = False
    return None


def resample_to_weekly(df):
    """
    将日度收益率数据聚合为周度（按周五划分）。
    日度收益率 → 周度复利累计收益率。

    参数:
        df: DataFrame，必须包含列 ['date', 'portfolio_return', 'bench_return', 'excess']
            （或 'long_return' 替代 'portfolio_return'）

    返回:
        DataFrame，结构与输入一致，按周聚合
    """
    # 确定收益率列名
    ret_col = 'portfolio_return' if 'portfolio_return' in df.columns else 'long_return'

    df = df.copy()
    # 确保 date 为 datetime 并设为 index
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    # W-FRI: 以周五为周边界进行重采样
    weekly = df.resample('W-FRI').agg({
        ret_col: lambda x: (1 + x).prod() - 1,
        'bench_return': lambda x: (1 + x).prod() - 1,
        'excess': lambda x: (1 + x).prod() - 1,
    })
    weekly = weekly.reset_index()
    # 修正 resample 后的日期为当周最后一个实际交易日
    # （保留原始最后一天的日期更准确）
    return weekly


def plot_excess_curve(df, title, filename, color_palette=None, output_dir=None, freq='daily'):
    """
    绘制超额收益曲线图。
    上子图：累计收益率曲线（持仓 vs 基准 vs 超额）
    下子图：每日/每周超额收益柱状图

    参数:
        freq: 'daily' 或 'weekly'，影响坐标轴标签和统计文字
    """
    is_weekly = freq == 'weekly'
    period_label = '周' if is_weekly else '日'
    period_name = '周度' if is_weekly else '每日'
    if df.empty:
        print(f"[WARN] 无数据可绘图: {title}")
        return

    sns.set_style("whitegrid")
    # 必须在 sns.set_style 之后设置字体，否则 seaborn 会覆盖 font.sans-serif
    font_name = set_chinese_font()

    if color_palette is None:
        color_palette = ['#2E86AB', '#A23B72', '#F18F01']

    fig = plt.figure(figsize=(16, 10), facecolor='#FAFAFA')
    fig.suptitle(title, fontsize=20, fontweight='bold',
                 y=0.97, color='#2C3E50')

    # ---- 布局 ----
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.2, 0.08],
                          hspace=0.25, left=0.08, right=0.92, top=0.90, bottom=0.08)

    # ===== 上子图：累计收益率曲线 =====
    ax1 = fig.add_subplot(gs[0, 0])

    dates = df['date'].values
    dates_num = mdates.date2num(dates)

    # 计算累计收益率（cumprod）
    cum_long = (1 + df['long_return'].fillna(0)).cumprod() - 1 \
        if 'long_return' in df.columns else None
    cum_portfolio = (1 + df['portfolio_return'].fillna(0)).cumprod() - 1 \
        if 'portfolio_return' in df.columns else None
    cum_bench = (1 + df['bench_return'].fillna(0)).cumprod() - 1
    cum_excess = (1 + df['excess'].fillna(0)).cumprod() - 1

    plot_data = []
    if cum_long is not None:
        plot_data.append(('多头收益率', cum_long, color_palette[0], '-'))
    if cum_portfolio is not None:
        plot_data.append(('组合收益率', cum_portfolio, color_palette[0], '-'))
    plot_data.append(('基准收益率', cum_bench, color_palette[1], '--'))
    plot_data.append(('累计超额', cum_excess, color_palette[2], '-'))

    for label, data, color, style in plot_data:
        if data is not None:
            ax1.plot(dates_num, data * 100, label=label, color=color,
                     linewidth=2.2 if style == '-' else 1.8,
                     linestyle=style, alpha=0.9)

    # 零线
    ax1.axhline(y=0, color='#888888', linewidth=0.8, linestyle=':', alpha=0.7)
    ax1.fill_between(dates_num, 0, cum_excess * 100,
                     where=(cum_excess >= 0),
                     color=color_palette[2], alpha=0.08, interpolate=True)
    ax1.fill_between(dates_num, 0, cum_excess * 100,
                     where=(cum_excess < 0),
                     color='#D1495B', alpha=0.08, interpolate=True)

    ax1.set_ylabel('累计收益率 (%)', fontsize=13, color='#2C3E50')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True,
               fontsize=11, edgecolor='#CCCCCC')
    # x 轴格式：周度显示日期，日度显示年月
    if is_weekly:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=4))  # 每2周的周五
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.set_title('累计收益率对比', fontsize=14, fontweight='bold',
                  color='#2C3E50', pad=12)

    # 自动调整 y 轴范围
    all_vals = [d * 100 for _, d, _, _ in plot_data if d is not None]
    if all_vals:
        all_flat = np.concatenate([v for v in all_vals if v is not None])
        if len(all_flat) > 0:
            ymin = np.nanmin(all_flat)
            ymax = np.nanmax(all_flat)
            ypad = max(abs(ymax - ymin) * 0.15, 1)
            ax1.set_ylim(ymin - ypad, ymax + ypad)

    # ===== 下子图：每日/每周超额柱状图 =====
    ax2 = fig.add_subplot(gs[1, 0])

    excess_daily = df['excess'].values * 100
    colors_bar = [color_palette[2] if v >= 0 else '#D1495B' for v in excess_daily]
    bar_width = 0.6 if is_weekly else 0.8
    ax2.bar(dates_num, excess_daily, width=bar_width, color=colors_bar,
            alpha=0.7, edgecolor='white', linewidth=0.3)

    ax2.axhline(y=0, color='#888888', linewidth=0.8)
    ax2.set_ylabel(f'{period_label}超额收益 (%)', fontsize=13, color='#2C3E50')
    # x 轴格式同步
    if is_weekly:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2, byweekday=4))
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.tick_params(axis='y', labelsize=10)

    # 统计信息标注
    pos_count = sum(1 for v in excess_daily if v > 0)
    neg_count = sum(1 for v in excess_daily if v < 0)
    total_count = len(excess_daily)
    win_rate = pos_count / total_count * 100 if total_count > 0 else 0
    avg_excess = np.nanmean(excess_daily)
    total_excess_cum = cum_excess.iloc[-1] * 100 if len(cum_excess) > 0 else 0

    stats_text = (
        f"胜率: {win_rate:.1f}%  "
        f"{period_label}均超额: {avg_excess:.3f}%  "
        f"累计超额: {total_excess_cum:.2f}%  "
        f"交易{period_label}: {total_count}"
    )
    ax2.text(0.5, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0',
                       edgecolor='#CCCCCC', alpha=0.9))

    ax2.set_title(f'{period_name}超额收益分布', fontsize=14, fontweight='bold',
                  color='#2C3E50', pad=12)

    # 调整 y 轴
    daily_vals = excess_daily[~np.isnan(excess_daily)]
    if len(daily_vals) > 0:
        dymax = np.max(np.abs(daily_vals))
        dypad = max(dymax * 0.2, 0.5)
        ax2.set_ylim(-dymax - dypad, dymax + dypad)

    # ===== 页脚 =====
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax3.text(0.5, 0.3, f'生成时间: {now_str}  |  数据源: 米筐 rqdatac + 本地持仓',
             transform=ax3.transAxes, ha='center', fontsize=9,
             color='#999999')

    # 保存
    save_dir = output_dir or CONFIG['output_dir']
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, filename)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"图表已保存: {output_path}")


def main():
    args = parse_args()
    freq = args.freq
    is_weekly = freq == 'weekly'
    freq_suffix = '_weekly' if is_weekly else ''

    output_dir = args.output or CONFIG['output_dir']
    CONFIG['output_dir'] = output_dir

    # 创建子目录
    data_dir = os.path.join(output_dir, 'data')
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    freq_name = '周度' if is_weekly else '日度'
    print("=" * 65)
    print("  超额收益分析工具")
    print(f"  项目目录: {output_dir}")
    print(f"  数据频率: {freq_name}")
    print("=" * 65)

    # ---- Step 0: 获取日期范围 ----
    print("\n[Step 0] 确定日期范围...")
    common_dates, actual_all, target_all = get_common_trading_dates(args.start, args.end)
    if not common_dates:
        print("[ERROR] 无有效日期，退出")
        sys.exit(1)
    if not os.path.exists(os.path.join(data_dir, f'{args.start}_{args.end}_actual_excess{freq_suffix}.csv')):

        # ---- Step 1: 实际持仓 ----
        print(f"\n{'='*65}")
        print("[Step 1] 加载实际持仓数据...")
        print(f"{'='*65}")
        actual_df = load_actual_data(common_dates)
        if not actual_df.empty:
            # 周度聚合
            if is_weekly:
                print("\n  聚合为周度数据...")
                plot_actual = resample_to_weekly(
                    actual_df.rename(columns={'long_return': 'portfolio_return'})
                )
                print(f"  周度数据: {len(plot_actual)} 行")
            else:
                plot_actual = actual_df.rename(columns={'long_return': 'portfolio_return'})

            # 保存数据到 data/
            actual_csv = os.path.join(data_dir, f'{args.start}_{args.end}_actual_excess{freq_suffix}.csv')
            plot_actual.to_csv(actual_csv, index=False, encoding='utf-8-sig',
                               float_format='%.8f')
            print(f"数据已保存: {actual_csv}")

            # 绘图到 charts/
            print("\n绘制实际持仓超额曲线图...")
            plot_excess_curve(
                plot_actual,
                title=f'实际持仓超额收益分析 — 海通PBHSZX1H ({freq_name})',
                filename=f'{args.start}_{args.end}_actual_excess{freq_suffix}.png',
                output_dir=charts_dir,
                freq=freq,
            )
        else:
            print("[WARN] 实际持仓数据为空")

        # ---- Step 2: 理论持仓 ----
        print(f"\n{'='*65}")
        print("[Step 2] 加载理论持仓数据（含米筐行情）...")
        print(f"{'='*65}")
        theoretical_df = load_theoretical_data(common_dates)
        if not theoretical_df.empty:
            # 周度聚合
            plot_theoretical = theoretical_df
            if is_weekly:
                print("\n  聚合为周度数据...")
                plot_theoretical = resample_to_weekly(theoretical_df)
                print(f"  周度数据: {len(plot_theoretical)} 行")

            # 保存数据到 data/
            theo_csv = os.path.join(data_dir, f'{args.start}_{args.end}_theoretical_excess{freq_suffix}.csv')
            plot_theoretical.to_csv(theo_csv, index=False, encoding='utf-8-sig',
                                    float_format='%.8f')
            print(f"数据已保存: {theo_csv}")

            # 绘图到 charts/
            print("\n绘制理论持仓超额曲线图...")
            plot_excess_curve(
                plot_theoretical,
                title=f'理论持仓超额收益分析 — TCHMD (基准: 沪深300) ({freq_name})',
                filename=f'{args.start}_{args.end}_theoretical_excess{freq_suffix}.png',
                output_dir=charts_dir,
                freq=freq,
            )
        else:
            print("[WARN] 理论持仓数据为空")

    manager = EmailManager()
    manager.send_email_with_images_and_attachments(['pagududeshengjiang@shpbjj.com', '961435548@qq.com'],
                                                   '向韩总报告-超额周度报告',
                                                   body_html=f'''                                                                                                                                                                                    
                                                            <h2>超额周度报告</h2>                                                                                                                                                                            
                                                            <p>周期: {common_dates[0]} ~ {common_dates[-1]}</p>                                                                                                                                    
                                                            <img src="cid:image_0" style="max-width:100%;">                                                                                                                                                  
                                                            <hr>                                                                                                                                                                                          
                                                            <p>详情见附件</p>                                                                                                                                                                     
                                                            ''',
                                                   image_paths=[os.path.join(charts_dir, f'{args.start}_{args.end}_theoretical_excess{freq_suffix}.png')],
                                                   attachments=[os.path.join(data_dir, f'{args.start}_{args.end}_theoretical_excess{freq_suffix}.csv')])


    # ---- 汇总 ----
    print(f"\n{'='*65}")
    print("  完成！")
    print(f"{'='*65}")
    print(f"项目目录: {output_dir}")
    print(f"  实际数据: data/{args.start}_{args.end}_actual_excess{freq_suffix}.csv")
    print(f"  理论数据: data/{args.start}_{args.end}_theoretical_excess{freq_suffix}.csv")
    print(f"  实际图表: charts/{args.start}_{args.end}_actual_excess{freq_suffix}.png")
    print(f"  理论图表: charts/{args.start}_{args.end}_theoretical_excess{freq_suffix}.png")
    print(f"\n提示: 使用 --freq weekly 切换为周度模式")


if __name__ == '__main__':
    main()
