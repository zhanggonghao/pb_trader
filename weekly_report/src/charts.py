"""Charts Generation Module"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Optional, Dict, List
from calculator import AttributionResult, StyleExposureResult

# ---- Theme Colors ----
COLOR_PRIMARY = '#1a3a5c'
COLOR_SECONDARY = '#4a90d9'
COLOR_POS = '#16a34a'
COLOR_NEG = '#e31b23'
COLOR_BENCH = '#f39c12'
COLOR_HEDGED = '#8e44ad'
COLOR_BG = '#f8f9fa'
COLOR_GRID = '#e0e0e0'


def _setup_style():
    """Set matplotlib style for professional look"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.facecolor': COLOR_BG,
        'axes.edgecolor': '#cccccc',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.color': COLOR_GRID,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })


def _pct_fmt():
    return plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%')


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ================================================================
#  Chart 1: Net Value Curves
# ================================================================
def plot_netvalue_curves(res: AttributionResult, out_path: str) -> str:
    _setup_style()
    _ensure_dir(out_path)
    daily = res.daily.copy()
    x = list(range(len(daily)))
    labels = [d[-5:] if isinstance(d, str) else pd.Timestamp(d).strftime('%m-%d') for d in daily['date']]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)

    ax.plot(x, daily['portfolio_nv'], color=COLOR_PRIMARY, linewidth=2.5,
            marker='o', markersize=6, label=f'组合虚拟净值')
    ax.plot(x, daily['benchmark_nv'], color=COLOR_BENCH, linewidth=2.0,
            marker='s', markersize=5, label=f'基准 {res.benchmark}')
    ax.plot(x, daily['hedged_nv'], color=COLOR_HEDGED, linewidth=2.0,
            marker='^', markersize=5, linestyle='--', label='对冲后净值')
    if 'nav_nv' in daily.columns:
        ax.plot(x, daily['nav_nv'], color=COLOR_POS, linewidth=2.2,
                marker='D', markersize=5, label='产品实际净值')

    # Fill between portfolio and benchmark
    ax.fill_between(x, daily['benchmark_nv'], daily['portfolio_nv'],
                    alpha=0.08, color=COLOR_POS, label='超额区域')

    ax.set_title(f'{res.product_name} 净值走势', fontsize=14, fontweight='bold',
                 color=COLOR_PRIMARY, pad=15)
    ax.set_xlabel('交易日', fontsize=11)
    ax.set_ylabel('累计净值（初始=1）', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.yaxis.set_major_formatter(_pct_fmt())
    ax.legend(loc='best', framealpha=0.9, fontsize=9, edgecolor='#cccccc')
    ax.set_ylim(bottom=min(daily[['portfolio_nv', 'benchmark_nv', 'hedged_nv']].min()) * 0.995)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 2: Daily Attribution Bar Chart
# ================================================================
def plot_daily_attribution(res: AttributionResult, out_path: str) -> str:
    _setup_style()
    _ensure_dir(out_path)
    daily = res.daily.copy()
    x = np.arange(len(daily))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=200)

    bars1 = ax.bar(x - w, daily['portfolio_return'], width=w, label='组合收益',
                   color=COLOR_PRIMARY, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, daily['benchmark_return'], width=w, label='基准收益',
                   color=COLOR_BENCH, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + w, daily['excess_return'], width=w, label='超额收益',
                   color=[COLOR_POS if v >= 0 else COLOR_NEG for v in daily['excess_return']],
                   alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    labels = [d[-5:] if isinstance(d, str) else pd.Timestamp(d).strftime('%m-%d') for d in daily['date']]
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(_pct_fmt())
    ax.set_ylabel('日收益', fontsize=11)
    ax.set_title('日度收益分解', fontsize=14, fontweight='bold', color=COLOR_PRIMARY, pad=15)
    ax.legend(loc='best', framealpha=0.9, fontsize=9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 3: Stock Contribution (Top + Bottom)
# ================================================================
def plot_stock_contribution(res: AttributionResult, out_path: str,
                            top_n: int = 10, name_lookup: Optional[Dict] = None) -> str:
    _setup_style()
    _ensure_dir(out_path)

    sc = res.stock_contrib
    if sc.empty:
        return out_path

    top = sc.head(top_n).copy()
    bot = sc.tail(top_n).iloc[::-1].copy()
    combined = pd.concat([top, bot], ignore_index=True)
    combined['display'] = combined['order_book_id']
    if name_lookup:
        combined['display'] = combined['order_book_id'].map(
            lambda c: f'{name_lookup.get(c, c)} ({c})'
        )
    combined = combined.sort_values('contrib', ascending=True)

    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in combined['contrib']]
    n_stocks = len(combined)
    fig_height = max(4, 0.35 * n_stocks)

    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=200)

    bars = ax.barh(combined['display'], combined['contrib'], color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    # Add value labels
    for bar, val in zip(bars, combined['contrib']):
        if val >= 0:
            ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                    f'+{val*100:.2f}%', va='center', fontsize=8, color=COLOR_POS)
        else:
            ax.text(bar.get_width() - 0.0001, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.2f}%', va='center', ha='right', fontsize=8, color=COLOR_NEG)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('整周累计贡献（占组合权重 x 个股期间收益）', fontsize=10)
    ax.set_title(f'个股贡献排行（正负各 Top {top_n}）', fontsize=14,
                 fontweight='bold', color=COLOR_PRIMARY, pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 4: Style Exposure Radar
# ================================================================
def plot_style_radar(style: StyleExposureResult, out_path: str,
                     clip: float = 3.0) -> str:
    _setup_style()
    _ensure_dir(out_path)

    n = len(style.factors)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # Get active exposure values
    if not style.active_exposure.empty:
        raw_vals = style.active_exposure.iloc[0].values
    else:
        raw_vals = np.zeros(n)

    vals = np.clip(raw_vals, -clip, clip)
    vals_closed = vals.tolist() + vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=200, subplot_kw=dict(polar=True))

    ax.plot(angles_closed, vals_closed, 'o-', color=COLOR_PRIMARY, linewidth=2.5,
            markersize=7, label='主动偏离')
    ax.fill(angles_closed, vals_closed, color=COLOR_PRIMARY, alpha=0.12)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

    # Labels with z-score
    label_texts = []
    z_dict = style.z_scores.to_dict() if hasattr(style.z_scores, 'to_dict') else {}
    for f, label in zip(style.factors, style.factor_labels):
        z = z_dict.get(f, 0)
        if abs(z) > 0.5:
            label_texts.append(f'{label}
(z={z:+.2f})')
        else:
            label_texts.append(label)

    ax.set_xticks(angles)
    ax.set_xticklabels(label_texts, fontsize=9)
    ax.set_ylim(-clip, clip)
    ax.set_yticks([-clip, -clip/2, 0, clip/2, clip])
    ax.set_yticklabels([f'{int(-clip)}s', f'{int(-clip/2)}s', '0',
                        f'{int(clip/2)}s', f'{int(clip)}s'], fontsize=8)
    ax.set_title('风格敞口暴露雷达图（组合 vs 基准的主动偏离）',
                 fontsize=13, fontweight='bold', color=COLOR_PRIMARY, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 5: Style Exposure Horizontal Bars
# ================================================================
def plot_style_bars(style: StyleExposureResult, out_path: str,
                    clip: float = 3.0) -> str:
    _setup_style()
    _ensure_dir(out_path)

    if style.active_exposure.empty:
        return out_path

    raw_vals = style.active_exposure.iloc[0].values
    z_dict = style.z_scores.to_dict() if hasattr(style.z_scores, 'to_dict') else {}

    plotted_vals = np.clip(raw_vals, -clip, clip)
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in raw_vals]

    labels = []
    for f, label in zip(style.factors, style.factor_labels):
        z = z_dict.get(f, 0)
        if abs(z) > 0.5:
            labels.append(f'{label}  (z={z:+.2f})')
        else:
            labels.append(label)

    sorted_idx = np.argsort(plotted_vals)
    sorted_vals = plotted_vals[sorted_idx]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    bars = ax.barh(sorted_labels, sorted_vals, color=sorted_colors,
                   edgecolor='white', linewidth=0.5, height=0.6)

    # Add value labels
    for bar, val in zip(bars, sorted_vals):
        if val >= 0:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'+{val:.3f}', va='center', fontsize=8, color=COLOR_POS)
        else:
            ax.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='right', fontsize=8, color=COLOR_NEG)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlim(-clip * 1.15, clip * 1.15)
    ax.set_xlabel('主动偏离度（正值 = 组合比基准更偏好该风格）', fontsize=10)
    ax.set_title('风格敞口暴露（主动偏离排序）', fontsize=13,
                 fontweight='bold', color=COLOR_PRIMARY, pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 6: Industry Distribution (Pie)
# ================================================================
def plot_industry_distribution(industry_df: pd.DataFrame, out_path: str,
                               title: str = '持仓行业分布') -> str:
    _setup_style()
    _ensure_dir(out_path)

    if industry_df.empty:
        return out_path

    # Take last date's industry distribution
    last_date = industry_df['date'].max()
    ind_data = industry_df[industry_df['date'] == last_date].copy()
    ind_data = ind_data.sort_values('weight', ascending=False)

    # Group small ones
    threshold = 0.03
    main = ind_data[ind_data['weight'] >= threshold].copy()
    other_sum = ind_data[ind_data['weight'] < threshold]['weight'].sum()
    if other_sum > 0:
        main = pd.concat([main, pd.DataFrame([{'industry': '其他', 'weight': other_sum, 'date': last_date}])],
                         ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    colors = plt.cm.Set3(np.linspace(0, 1, len(main)))

    wedges, texts, autotexts = ax.pie(
        main['weight'], labels=main['industry'], autopct='%1.1f%%',
        colors=colors, startangle=90,
        textprops={'fontsize': 9},
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    )

    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold', color=COLOR_PRIMARY, pad=20)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 7: Concentration Bar
# ================================================================
def plot_concentration(concentration: Dict, out_path: str, top_n: int = 10) -> str:
    _setup_style()
    _ensure_dir(out_path)

    dates = sorted(concentration.keys())
    weights = [concentration[d]['top_n_weight'] for d in dates]
    labels = [d[-5:] if len(d) > 5 else d for d in dates]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    bars = ax.bar(range(len(dates)), weights, color=COLOR_PRIMARY, alpha=0.85,
                  edgecolor='white', linewidth=0.5, width=0.5)

    for bar, val in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=COLOR_PRIMARY)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('权重占比', fontsize=11)
    ax.set_title(f'持仓集中度（Top {top_n} 权重）', fontsize=14,
                 fontweight='bold', color=COLOR_PRIMARY, pad=15)
    ax.set_ylim(0, max(weights) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ================================================================
#  Chart 8: Style Exposure Time Series
# ================================================================
def plot_style_timeseries(style: StyleExposureResult, out_path: str) -> str:
    _setup_style()
    _ensure_dir(out_path)

    if style.daily_portfolio.empty or style.daily_benchmark.empty:
        return out_path

    n = len(style.factors)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows), dpi=200)
    axes = axes.flatten()

    x_idx = list(range(len(style.daily_portfolio)))
    x_labels = [str(d)[-5:] for d in style.daily_portfolio.index]

    for i, (f, label) in enumerate(zip(style.factors, style.factor_labels)):
        ax = axes[i]
        if f in style.daily_portfolio.columns and f in style.daily_benchmark.columns:
            port_vals = style.daily_portfolio[f].values
            bench_vals = style.daily_benchmark[f].values

            ax.plot(x_idx, port_vals, 'o-', color=COLOR_PRIMARY, linewidth=2.0,
                    markersize=6, label='组合')
            ax.plot(x_idx, bench_vals, 's--', color=COLOR_BENCH, linewidth=1.5,
                    markersize=5, label='基准')

            ax.fill_between(x_idx, bench_vals, port_vals, alpha=0.1, color=COLOR_SECONDARY)

        ax.set_title(label, fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle('风格因子每日暴露变化', fontsize=14, fontweight='bold',
                 color=COLOR_PRIMARY, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
