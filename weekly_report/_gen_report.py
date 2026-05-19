"""Core Calculation Module: performance attribution, style exposure"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AttributionResult:
    """收益归因结果"""
    product_code: str
    product_name: str
    benchmark: str
    period: str
    start_date: str
    end_date: str
    daily: pd.DataFrame  # 日度数据
    stock_contrib: pd.DataFrame  # 个股贡献
    summary: dict  # 汇总指标


@dataclass
class StyleExposureResult:
    """风格暴露结果"""
    factors: List[str]
    factor_labels: List[str]
    portfolio_exposure: pd.DataFrame  # 组合暴露
    benchmark_exposure: pd.DataFrame  # 基准暴露
    active_exposure: pd.DataFrame  # 主动偏离
    z_scores: pd.Series  # z-score
    daily_portfolio: pd.DataFrame  # 每日组合暴露时序
    daily_benchmark: pd.DataFrame  # 每日基准暴露时序


def calculate_stock_daily_returns(
    positions: Dict[str, pd.DataFrame],
    close_prices: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """计算每日持仓个股收益
    
    对每个交易日，合并持仓和收盘价，计算当日个股收益
    """
    stock_rets = {}
    prev_prices = None
    dates = sorted(positions.keys())
    
    for i, date in enumerate(dates):
        if date not in close_prices:
            continue
        pos = positions[date].copy()
        prices = close_prices[date][['code', 'close']].copy()
        prices.columns = ['code', 'close']
        
        merged = pos.merge(prices, on='code', how='left')
        
        if prev_prices is not None:
            # 计算个股日收益
            merged_prev = merged.merge(
                prev_prices[['code', 'close']], on='code', how='left',
                suffixes=('', '_prev')
            )
            merged_prev['stock_return'] = merged_prev['close'] / merged_prev['close_prev'] - 1
            merged_prev['stock_return'] = merged_prev['stock_return'].fillna(0)
            stock_rets[date] = merged_prev
        else:
            # 第一天没有前日收盘价，收益设为0
            merged['stock_return'] = 0.0
            stock_rets[date] = merged
        
        prev_prices = prices
    
    return stock_rets


def calculate_portfolio_daily_return(
    stock_rets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """计算组合每日收益"""
    records = []
    for date, df in stock_rets.items():
        port_ret = (df['weight'] * df['stock_return']).sum()
        records.append({'date': date, 'portfolio_return': port_ret})
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


def calculate_benchmark_daily_return(
    close_prices: Dict[str, pd.DataFrame],
    bench_weights: pd.DataFrame,
    trading_dates: List[str]
) -> pd.DataFrame:
    """计算基准每日收益"""
    if bench_weights is None:
        return pd.DataFrame(columns=['date', 'benchmark_return'])
    
    records = []
    prev_prices = None
    
    for date in trading_dates:
        if date not in close_prices:
            continue
        prices = close_prices[date][['code', 'close']].copy()
        prices['code'] = prices['code'].astype(str)
        
        merged = bench_weights.merge(prices, left_on='code_clean', right_on='code', how='left')
        
        if prev_prices is not None:
            merged_prev = merged.merge(
                prev_prices[['code', 'close']], on='code', how='left',
                suffixes=('', '_prev')
            )
            merged_prev['stock_return'] = merged_prev['close'] / merged_prev['close_prev'] - 1
            merged_prev['stock_return'] = merged_prev['stock_return'].fillna(0)
            bench_ret = (merged_prev['w'] * merged_prev['stock_return']).sum()
        else:
            bench_ret = 0.0
        
        records.append({'date': date, 'benchmark_return': bench_ret})
        prev_prices = prices
    
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


def calculate_daily_nav_return(nav_df: pd.DataFrame) -> pd.DataFrame:
    """计算产品净值日收益"""
    if nav_df.empty:
        return pd.DataFrame(columns=['date', 'nav_return'])
    df = nav_df.copy()
    df['nav_return'] = df['nav'].pct_change().fillna(0)
    return df


def attribution(
    positions: Dict[str, pd.DataFrame],
    close_prices: Dict[str, pd.DataFrame],
    nav_df: pd.DataFrame,
    bench_weights: Optional[pd.DataFrame],
    trading_dates: List[str],
    product_code: str,
    product_name: str,
    benchmark: str,
    period: str,
    start_date: str,
    end_date: str
) -> AttributionResult:
    """主归因函数"""
    # 个股日收益
    stock_rets = calculate_stock_daily_returns(positions, close_prices)
    
    # 组合日收益
    port_daily = calculate_portfolio_daily_return(stock_rets)
    
    # 基准日收益
    bench_daily = calculate_benchmark_daily_return(close_prices, bench_weights, trading_dates)
    
    # 净值日收益
    nav_daily = calculate_daily_nav_return(nav_df)
    
    # 合并日度数据
    daily = port_daily.copy()
    if not bench_daily.empty:
        daily = daily.merge(bench_daily, on='date', how='left')
        daily['benchmark_return'] = daily['benchmark_return'].fillna(0)
    else:
        daily['benchmark_return'] = 0.0
    
    daily['excess_return'] = daily['portfolio_return'] - daily['benchmark_return']
    
    if not nav_daily.empty:
        daily = daily.merge(nav_daily[['date', 'nav_return']], on='date', how='left')
        daily['nav_return'] = daily['nav_return'].fillna(0)
    else:
        daily['nav_return'] = 0.0
    
    # 累计净值
    daily['portfolio_nv'] = (1 + daily['portfolio_return']).cumprod()
    daily['benchmark_nv'] = (1 + daily['benchmark_return']).cumprod()
    daily['hedged_nv'] = (1 + daily['excess_return']).cumprod()
    if not nav_daily.empty:
        daily['nav_nv'] = (1 + daily['nav_return']).cumprod()
    else:
        daily['nav_nv'] = 1.0
    
    # 个股区间贡献
    stock_contrib_list = []
    for date, df in stock_rets.items():
        if 'stock_return' in df.columns:
            df_temp = df[['code', 'weight', 'stock_return']].copy()
            df_temp['date'] = date
            stock_contrib_list.append(df_temp)
    
    if stock_contrib_list:
        all_stock = pd.concat(stock_contrib_list, ignore_index=True)
        stock_contrib = all_stock.groupby('code').apply(
            lambda g: (g['weight'] * (1 + g['stock_return'])).prod() - g['weight'].iloc[0]
        ).reset_index()
        stock_contrib.columns = ['order_book_id', 'contrib']
        stock_contrib = stock_contrib.sort_values('contrib', ascending=False).reset_index(drop=True)
    else:
        stock_contrib = pd.DataFrame(columns=['order_book_id', 'contrib'])
    
    # 汇总指标
    summary = {
        'portfolio_return': daily['portfolio_return'].sum(),
        'benchmark_return': daily['benchmark_return'].sum(),
        'excess_return': daily['excess_return'].sum(),
        'nav_return': daily['nav_return'].sum() if 'nav_return' in daily.columns else 0,
        'portfolio_nv': daily['portfolio_nv'].iloc[-1] if len(daily) > 0 else 1,
        'benchmark_nv': daily['benchmark_nv'].iloc[-1] if len(daily) > 0 else 1,
        'max_drawdown': _max_drawdown(daily['portfolio_nv'].values) if len(daily) > 0 else 0,
        'trading_days': len(daily),
        'avg_position_count': _avg_position_count(positions),
    }
    
    return AttributionResult(
        product_code=product_code,
        product_name=product_name,
        benchmark=benchmark,
        period=period,
        start_date=start_date,
        end_date=end_date,
        daily=daily,
        stock_contrib=stock_contrib,
        summary=summary,
    )


def _max_drawdown(nv_series: np.ndarray) -> float:
    """计算最大回撤"""
    peak = np.maximum.accumulate(nv_series)
    drawdown = (nv_series - peak) / peak
    return float(np.min(drawdown))


def _avg_position_count(positions: Dict) -> float:
    """平均持仓数量"""
    counts = [len(df) for df in positions.values()]
    return np.mean(counts) if counts else 0


def analyze_style_exposure(
    positions: Dict[str, pd.DataFrame],
    factor_data_dict: Dict[str, pd.DataFrame],
    bench_weights: Optional[pd.DataFrame],
    trading_dates: List[str],
    factor_config: List[dict],
    history_window: int = 252
) -> Optional[StyleExposureResult]:
    """风格暴露分析"""
    factor_names = [f['name'] for f in factor_config]
    factor_labels = [f['label'] for f in factor_config]
    
    if not factor_data_dict:
        print('Warning: No factor data available')
        return None
    
    # 每日组合暴露
    daily_port_exp = pd.DataFrame(index=trading_dates, columns=factor_names, dtype=float)
    daily_bench_exp = pd.DataFrame(index=trading_dates, columns=factor_names, dtype=float)
    
    for date in trading_dates:
        if date not in factor_data_dict or date not in positions:
            continue
        
        pos = positions[date]
        factors = factor_data_dict[date]
        factors['code'] = factors['code'].astype(str).str.strip()
        
        # 组合暴露
        merged = pos.merge(factors, on='code', how='left')
        for fn in factor_names:
            if fn in merged.columns:
                daily_port_exp.loc[date, fn] = (merged['weight'] * merged[fn]).sum()
        
        # 基准暴露
        if bench_weights is not None:
            bench_merged = bench_weights.merge(factors, left_on='code_clean', right_on='code', how='left')
            for fn in factor_names:
                if fn in bench_merged.columns:
                    daily_bench_exp.loc[date, fn] = (bench_merged['w'] * bench_merged[fn]).sum()
    
    daily_port_exp = daily_port_exp.dropna(how='all')
    daily_bench_exp = daily_bench_exp.dropna(how='all')
    
    if daily_port_exp.empty:
        return None
    
    # 区间平均暴露
    port_mean = daily_port_exp.mean()
    bench_mean = daily_bench_exp.mean() if not daily_bench_exp.empty else pd.Series(0, index=factor_names)
    active = port_mean - bench_mean
    
    # z-score
    z_scores = pd.Series(0.0, index=factor_names)
    for fn in factor_names:
        if fn in daily_bench_exp.columns:
            hist_std = daily_bench_exp[fn].std()
            if hist_std > 0:
                z_scores[fn] = active[fn] / hist_std
    
    return StyleExposureResult(
        factors=factor_names,
        factor_labels=factor_labels,
        portfolio_exposure=port_mean.to_frame('actual').T,
        benchmark_exposure=bench_mean.to_frame('benchmark').T,
        active_exposure=active.to_frame('active').T,
        z_scores=z_scores,
        daily_portfolio=daily_port_exp,
        daily_benchmark=daily_bench_exp,
    )


def calculate_concentration(
    positions: Dict[str, pd.DataFrame],
    top_n: int = 10
) -> Dict:
    """计算持仓集中度"""
    result = {}
    for date in sorted(positions.keys()):
        df = positions[date].sort_values('weight', ascending=False).head(top_n)
        result[date] = {
            'top_n_weight': df['weight'].sum(),
            'top_n_count': len(df),
        }
    return result


def calculate_industry_exposure(
    positions: Dict[str, pd.DataFrame],
    stock_industry: Dict[str, str],
    trading_dates: List[str]
) -> pd.DataFrame:
    """计算行业暴露"""
    records = []
    for date in trading_dates:
        if date not in positions:
            continue
        pos = positions[date].copy()
        pos['industry'] = pos['code'].map(stock_industry).fillna('Other')
        ind_exp = pos.groupby('industry')['weight'].sum().reset_index()
        ind_exp['date'] = date
        records.append(ind_exp)
    
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()
