import os
import traceback

import pandas as pd
import numpy as np

import rqdatac
# rqdatac.init(13601611030,'PB123456789')
# rqdatac.init(username='18101949790', password='123456')
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)


from ultis.stock_data_client import StockDataClient
# from ultis import myrqfactor
import ultis.myrqfactor as myrqfactor



# 仅使用 myrqfactor 中真实存在、可被 add_factor 获取的因子。
# 这里采用“Barra 风格 -> myrqfactor 代理因子”的映射，与 PBackTest 保持一致。
BARRA_PROXY_FACTOR_SPECS = {
    'lcap_exposure': [('lcap', 1.0)],
    # 'srmi_exposure': [('srmi', 1.0)],
    'beta_exposure': [('ddnbt240', 1.0), ('uupbt240', 1.0)],
    'liquidity_exposure': [('illiquidity', -1.0)],
    # 'residual_volatility_exposure': [('ddnsr240', 1.0), ('uupsr240', 1.0), ('chvolatility', 1.0)],
}

LEGACY_ALIAS_MAP = {
    'size_exposure': 'lcap_exposure',
    # 'momentum_exposure': 'srmi_exposure',
    'non_linear_size_exposure': 'nl_size',
}

# myrqfactor.add_factor 在 ddnbt240 / uupbt240 上无法直接计算，统一手工实现
MANUAL_FACTOR_NAMES = {'ddnbt240', 'uupbt240'}

# 输出 stock_factors 时的列顺序（与 backtest/factor_explosure.py 对齐）
STOCK_OUTPUT_COLS = [
    'date',
    'order_book_id',
    'lcap_exposure',
    # 'srmi_exposure',
    'beta_exposure',
    'liquidity_exposure',
    # 'residual_volatility_exposure',
    # 'nl_size',
]


def _log(message: str):
    print(f'[factor_explosure] {message}', flush=True)


def _zscore_cs(x: pd.Series) -> pd.Series:
    """
    截面标准化
    """
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def _nl_size_one_day(g: pd.DataFrame) -> pd.Series:
    x = g['lcap'].astype(float)
    mask = x.notna()
    if mask.sum() < 2:
        return pd.Series(np.nan, index=g.index, name='nl_size_raw')

    x2 = x[mask]
    y = x2 ** 3
    X = np.column_stack([np.ones(len(x2)), x2.values])
    try:
        beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        y_hat = X @ beta
        resid = y.values - y_hat
        s = pd.Series(np.nan, index=g.index, name='nl_size_raw')
        s.loc[mask] = resid
        return s
    except Exception:
        return pd.Series(np.nan, index=g.index, name='nl_size_raw')


def compute_nl_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于横截面回归：对每个日期，y = lcap^3，X = [1, lcap]，残差取反并横截面标准化。
    返回含列：['date','order_book_id','nl_size']。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'order_book_id'])

    nl_raw = (
        df.groupby('date', group_keys=False)
          .apply(_nl_size_one_day)
    )
    df['nl_size_raw'] = nl_raw.values
    df['nl_size'] = (
        df.groupby('date', group_keys=False)['nl_size_raw']
          .transform(_zscore_cs)
    ) * -1.0

    return df[['date', 'order_book_id', 'nl_size']]


def _load_index_close(index_code: str, start, end, client: StockDataClient) -> pd.DataFrame:
    try:
        idx_close = rqdatac.get_price(
            order_book_ids=index_code,
            start_date=start,
            end_date=end,
            frequency="1d",
            fields=["close"],
        )
        if isinstance(idx_close, pd.Series):
            idx_close = idx_close.to_frame(name="close")
        elif "close" not in idx_close.columns and idx_close.shape[1] == 1:
            idx_close = idx_close.rename(columns={idx_close.columns[0]: "close"})
        idx_close = idx_close.reset_index()
        date_col = "date" if "date" in idx_close.columns else idx_close.columns[0]
        idx_close = idx_close.rename(columns={date_col: "date"})
        idx_close["date"] = pd.to_datetime(idx_close["date"])
        idx_close["order_book_id"] = index_code
        return idx_close[["date", "order_book_id", "close"]]
    except Exception:
        fallback = (
            client.get_stock_post_1d_data(order_book_ids=[index_code], start=start, end=end)["close"]
            .reset_index()
        )
        fallback["date"] = pd.to_datetime(fallback["date"])
        return fallback[["date", "order_book_id", "close"]]


def _compute_conditional_beta_panel(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int,
    side: str,
) -> pd.DataFrame:
    if side == 'down':
        cond = benchmark_returns < 0
    elif side == 'up':
        cond = benchmark_returns > 0
    else:
        raise ValueError(f'unsupported side={side}')

    x = benchmark_returns.where(cond)
    y = stock_returns.where(cond, np.nan)

    n = x.notna().astype(float).rolling(window, min_periods=1).sum()
    sum_x = x.fillna(0.0).rolling(window, min_periods=1).sum()
    sum_x2 = x.pow(2).fillna(0.0).rolling(window, min_periods=1).sum()
    sum_y = y.fillna(0.0).rolling(window, min_periods=1).sum()
    sum_xy = y.mul(x, axis=0).fillna(0.0).rolling(window, min_periods=1).sum()

    mean_x = sum_x / n
    mean_y = sum_y.div(n, axis=0)
    cov = sum_xy.div(n, axis=0) - mean_y.mul(mean_x, axis=0)
    var = sum_x2 / n - mean_x.pow(2)

    beta = cov.div(var, axis=0)
    beta = beta.where(n >= 2, np.nan)
    beta = beta.where(var > 0, np.nan)
    return beta


def build_manual_factor_panel(
    base_df: pd.DataFrame,
    client: StockDataClient,
    order_book_ids,
    benchmark_code,
    start,
    end,
    window: int = 252,
) -> pd.DataFrame:
    """
    手工计算 ddnbt240 / uupbt240：基于个股 close 收益率与 benchmark 收益率，
    分别在“基准下跌日”/“基准上涨日”做条件 beta（窗口 window）。
    """
    _log('开始手工计算: ddnbt240, uupbt240')
    manual = base_df[['date', 'order_book_id']].drop_duplicates().copy()
    manual['date'] = pd.to_datetime(manual['date'])

    _log(f'手工计算条件 beta: benchmark={benchmark_code}, window={window}')
    stock_close = (
        client.get_stock_post_1d_data(order_book_ids=list(order_book_ids), start=start, end=end)['close']
        .reset_index()
    )
    stock_close['date'] = pd.to_datetime(stock_close['date'])
    stock_close = stock_close.sort_values(['date', 'order_book_id'])
    stock_close_wide = stock_close.pivot(index='date', columns='order_book_id', values='close').sort_index()
    stock_returns = stock_close_wide.pct_change()

    benchmark_close = _load_index_close(index_code=benchmark_code, start=start, end=end, client=client)
    benchmark_close = benchmark_close.sort_values('date')
    benchmark_returns = benchmark_close.set_index('date')['close'].pct_change().reindex(stock_returns.index)

    ddnbt = _compute_conditional_beta_panel(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        window=window,
        side='down',
    )
    uupbt = _compute_conditional_beta_panel(
        stock_returns=stock_returns,
        benchmark_returns=benchmark_returns,
        window=window,
        side='up',
    )
    beta_panel = pd.concat(
        [
            ddnbt.stack(dropna=False).rename('ddnbt240'),
            uupbt.stack(dropna=False).rename('uupbt240'),
        ],
        axis=1,
    ).reset_index()
    beta_panel.columns = ['date', 'order_book_id', 'ddnbt240', 'uupbt240']
    beta_panel['date'] = pd.to_datetime(beta_panel['date'])

    manual = manual.merge(beta_panel, on=['date', 'order_book_id'], how='left')
    _log(
        '手工计算条件 beta 完成: '
        f'ddnbt240非空行数={int(manual["ddnbt240"].notna().sum())}, '
        f'uupbt240非空行数={int(manual["uupbt240"].notna().sum())}'
    )
    return manual


def load_index_weights_panel(
    client: StockDataClient,
    order_book_id: str,
    start: str,
    end: str = None,
    factor_name: str = 'lcap',
    linear_lcap: bool = False,
    cap_method: str = 'adjusted',
    benchmark: str = None,
) -> pd.DataFrame:
    """
    本地计算指定指数在区间内的成分股权重, 输出长表 [date, order_book_id, weight]。

    Parameters
    ----------
    order_book_id : 样本指数代码 (例如 'zz1800', '000300.XSHG')
    benchmark     : 可选基准指数代码 (例如 '000300.XSHG'); 默认 None
        - benchmark 为 None 或与 order_book_id 相同 → 单指数自身权重 (本地官方权重 / fallback)
        - benchmark != order_book_id → 混合口径 (生效仅当 cap_method='adjusted'):
              * (date, stock) 属于 benchmark 当日成分 → 取 benchmark 本地存储官方权重
              * 否则                                → adj_cap_i / Σ_{k∈benchmark} adj_cap_k
          (这正是 sample_range='zz1800' 时希望与官方基准 000300 完全对齐的口径,
          其中 ~300 只 benchmark 内股票走本地权重, 其余 ~1500 只走调整市值占比)
    cap_method:
      - 'adjusted' (默认): 上述本地权重 / 调整市值口径
      - 'lcap'    : 旧版 lcap/exp(lcap) 自身归一化口径 (忽略 benchmark 参数)
    end : 区间结束日, None 时默认取"今天", 与原接口行为一致。
    """
    if end is None:
        end = pd.Timestamp.today().normalize().strftime('%Y-%m-%d')
    return _compute_self_weight_panel(
        client=client,
        index_code=order_book_id,
        start=start,
        end=end,
        factor_name=factor_name,
        linear_lcap=linear_lcap,
        cap_method=cap_method,
        benchmark=benchmark,
    )


def add_factor_columns(base_df: pd.DataFrame, factor_names, manual_factor_panel=None):
    """
    依次在 base_df 上添加原始因子列。
    - 若 factor_name 在 MANUAL_FACTOR_NAMES 中，从 manual_factor_panel 直接 merge；
    - 否则调用 myrqfactor.add_factor 计算。
    返回 (df, used_names, failed_factors)。
    """
    df = base_df.copy()
    used_names = []
    failed_factors = []
    total = len(factor_names)
    for i, factor_name in enumerate(factor_names, start=1):
        _log(f'[{i}/{total}] 开始计算原始因子: {factor_name}')
        if manual_factor_panel is not None and factor_name in MANUAL_FACTOR_NAMES:
            if factor_name not in manual_factor_panel.columns:
                failed_factors.append(
                    {
                        'factor_name': factor_name,
                        'error_type': 'KeyError',
                        'error_message': f'手工因子结果缺少列 {factor_name}',
                    }
                )
                _log(f'[{i}/{total}] 手工因子缺失: {factor_name}')
                continue
            df = df.merge(
                manual_factor_panel[['date', 'order_book_id', factor_name]],
                on=['date', 'order_book_id'],
                how='left',
            )
            non_null = int(df[factor_name].notna().sum())
            _log(f'[{i}/{total}] 手工因子计算完成: {factor_name}, 非空行数={non_null}')
            used_names.append(factor_name)
            continue
        try:
            df = myrqfactor.add_factor(df, factor_name=factor_name)
            if factor_name not in df.columns:
                raise KeyError(f'add_factor 未返回列: {factor_name}')
            non_null = int(df[factor_name].notna().sum())
            _log(f'[{i}/{total}] 原始因子计算完成: {factor_name}, 非空行数={non_null}')
            used_names.append(factor_name)
        except Exception as exc:
            _log(f'[{i}/{total}] 原始因子计算失败: {factor_name}, err={exc!r}')
            _log(traceback.format_exc())
            failed_factors.append(
                {
                    'factor_name': factor_name,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )
            continue
    return df, used_names, failed_factors


def compute_rolling_beta_exposures(
    df: pd.DataFrame,
    specs,
    window: int = 252,
    ycol: str = 'return',
) -> pd.DataFrame:
    """
    按 BARRA_PROXY_FACTOR_SPECS 把多个原始因子（含符号）合成成一个综合因子，
    再对每只股票按 ycol 做窗口为 window 的滚动 beta（cov/var）。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)
    expo = df[['date', 'order_book_id']].copy()

    if ycol not in df.columns:
        raise KeyError(f'缺少收益列: {ycol}')

    for exposure_col, components in specs.items():
        _log(f'开始计算滚动暴露: {exposure_col}, components={components}, window={window}')
        component_cols = []
        for factor_name, sign in components:
            if factor_name not in df.columns:
                _log(f'  组件跳过: exposure={exposure_col}, factor={factor_name}, 原始因子未成功计算')
                continue
            comp_col = f'{exposure_col}__{factor_name}'
            try:
                df[comp_col] = pd.to_numeric(df[factor_name], errors='coerce') * float(sign)
                non_null = int(df[comp_col].notna().sum())
                _log(
                    f'  组件完成: exposure={exposure_col}, factor={factor_name}, '
                    f'sign={sign}, 非空行数={non_null}'
                )
                component_cols.append(comp_col)
            except Exception as exc:
                _log(
                    f'  组件失败: exposure={exposure_col}, factor={factor_name}, '
                    f'sign={sign}, err={exc!r}'
                )
                _log(traceback.format_exc())
                raise

        if not component_cols:
            _log(f'滚动暴露跳过: {exposure_col}, 无可用组件因子')
            expo[exposure_col] = np.nan
            continue

        synth_col = f'{exposure_col}__synth'
        df[synth_col] = df[component_cols].sum(axis=1, min_count=1)
        synth_non_null = int(df[synth_col].notna().sum())
        _log(f'  合成因子完成: exposure={exposure_col}, 非空行数={synth_non_null}')

        out = []
        try:
            for _, g in df.groupby('order_book_id', sort=False):
                y = pd.to_numeric(g[ycol], errors='coerce')
                x = pd.to_numeric(g[synth_col], errors='coerce')
                var = x.rolling(window, min_periods=window).var(ddof=0)
                cov = x.rolling(window, min_periods=window).cov(y)
                beta = cov / var
                beta[var <= 0] = np.nan
                out.append(beta)
            exposure_series = pd.concat(out).sort_index()
            expo[exposure_col] = exposure_series.values
            non_null = int(expo[exposure_col].notna().sum())
            _log(f'滚动暴露完成: {exposure_col}, 非空行数={non_null}')
        except Exception as exc:
            _log(f'滚动暴露计算失败: {exposure_col}, err={exc!r}')
            _log(traceback.format_exc())
            raise

    return expo


def weighted_exposure_by_date(df: pd.DataFrame, exposure_cols) -> pd.DataFrame:
    """
    每日按权重求加权平均，得到指数级别的当日风格暴露。
    """
    records = []
    for trade_date, g in df.groupby('date'):
        record = {'date': trade_date}
        for col in exposure_cols:
            valid = g[['weight', col]].dropna()
            if valid.empty:
                record[col] = np.nan
            else:
                record[col] = np.average(valid[col].astype(float), weights=valid['weight'].astype(float))
        records.append(record)
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


# ----------------------------------------------------------------------
# 个股基准权重 / 动态单票上限
# 算法严格对齐 PBackTest-dev1-V0915-opt/generate_index_weights.py:
#   - cap_method='adjusted' (默认):
#       free_float_ratio = free_circulation / total
#       weight_ratio     = LayeredTable(free_float_ratio)   # 分级靠档
#       adj_shares       = total × weight_ratio
#       adj_cap          = adj_shares × close               # close 取除权报价
#       weight_i         = (本地基准成分内 → 本地存储权重;
#                           否则 → adj_cap_i / Σ_{k∈benchmark} adj_cap_k)
#       max_weight       = max(weight × multiplier, fallback)
#   - cap_method='lcap' (回退):
#       weight_i = cap_i / sum(cap_k in bench_mark on date),  cap = exp(lcap)
# ----------------------------------------------------------------------
def _weight_ratio_from_free_float_pct(pct: float) -> float:
    """自由流通比例(0~100) → 加权比例(0~1), 分级靠档表与指数公司公布表一致。"""
    if pct is None or not np.isfinite(pct) or pct < 0:
        return np.nan
    if pct <= 15.0:
        return float(np.ceil(pct)) / 100.0
    if pct <= 20.0:
        return 0.20
    if pct <= 30.0:
        return 0.30
    if pct <= 40.0:
        return 0.40
    if pct <= 50.0:
        return 0.50
    if pct <= 60.0:
        return 0.60
    if pct <= 70.0:
        return 0.70
    if pct <= 80.0:
        return 0.80
    return 1.00


def _load_shares_panel(order_book_ids, start: str, end: str) -> pd.DataFrame:
    """
    通过 ``rqdatac.get_shares`` 拉取 ``total`` (总股本) 与 ``free_circulation``
    (自由流通股本), 并按 (order_book_id, date) 在日历轴上 ffill 到日频。

    返回长表: ['date', 'order_book_id', 'total', 'free_circulation']。
    """
    obids = sorted({str(x) for x in order_book_ids if x})
    if not obids:
        raise ValueError('_load_shares_panel: 输入股票列表为空')

    raw = rqdatac.get_shares(
        order_book_ids=obids,
        start_date=start,
        end_date=end,
        fields=['total', 'free_circulation'],
        expect_df=True,
        market='cn',
    )
    if raw is None or len(raw) == 0:
        raise ValueError(
            f'rqdatac.get_shares 在区间 [{start}, {end}] 内未返回任何数据, '
            f'股票数={len(obids)}'
        )

    raw = raw.reset_index()
    if 'date' not in raw.columns:
        date_col = next(
            (c for c in raw.columns if 'date' in c.lower() or c == 'index'), None
        )
        if date_col is None:
            raise ValueError(
                f'rqdatac.get_shares 返回结构异常, 找不到日期列, columns={list(raw.columns)}'
            )
        raw = raw.rename(columns={date_col: 'date'})

    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw[['date', 'order_book_id', 'total', 'free_circulation']].dropna(
        subset=['date', 'order_book_id']
    )
    raw = raw.sort_values(['order_book_id', 'date'])

    full_dates = pd.date_range(start=start, end=end, freq='D')
    out = (
        raw
        .set_index('date')
        .groupby('order_book_id', sort=False)[['total', 'free_circulation']]
        .apply(lambda df: df.reindex(full_dates).ffill())
    )
    out.index.set_names(['order_book_id', 'date'], inplace=True)
    out = out.reset_index().dropna(subset=['total'])
    return out


def _load_close_panel(order_book_ids, start: str, end: str) -> pd.DataFrame:
    """
    通过 ``rqdatac.get_price`` 拉取日频 **除权报价 (adjust_type='none')** 收盘价,
    输出长表 [date, order_book_id, close]。与 ``rqdatac.get_shares`` 返回的当期
    总股本 / 自由流通股本相乘后即得到与指数公司一致口径的调整流通市值。
    """
    obids = sorted({str(x) for x in order_book_ids if x})
    if not obids:
        raise ValueError('_load_close_panel: 输入股票列表为空')

    raw = rqdatac.get_price(
        order_book_ids=obids,
        start_date=start,
        end_date=end,
        frequency='1d',
        fields=['close'],
        adjust_type='none',
        skip_suspended=False,
        expect_df=True,
        time_slice=None,
        market='cn',
    )
    if raw is None or len(raw) == 0:
        raise ValueError(
            f'rqdatac.get_price 在区间 [{start}, {end}] 内未返回任何数据, '
            f'股票数={len(obids)}'
        )

    raw = raw.reset_index()
    date_col = (
        'date' if 'date' in raw.columns
        else ('datetime' if 'datetime' in raw.columns else None)
    )
    if date_col is None:
        raise ValueError(
            f'rqdatac.get_price 返回结构异常, 找不到日期列, columns={list(raw.columns)}'
        )
    raw = raw.rename(columns={date_col: 'date'})
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw[['date', 'order_book_id', 'close']].dropna(
        subset=['date', 'order_book_id', 'close']
    )
    raw = raw[raw['close'] > 0]
    return raw


def _load_local_benchmark_weights(
    client: StockDataClient,
    benchmark: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    从本地数据库加载 ``benchmark`` 指数的成分股权重, 输出长表
    [date, order_book_id, bench_local_weight]。

    底层调 ``StockDataClient.get_stock_index_comments_weights_industry``,
    与回测/实盘端读取基准权重的入口完全一致。权重原值直接使用, 不做归一化。
    """
    bench_df = client.get_stock_index_comments_weights_industry(
        order_book_id=benchmark, start=start, end=end,
    )
    if bench_df is None or len(bench_df) == 0:
        raise ValueError(
            f'本地 benchmark 权重加载为空: {benchmark}, 区间=[{start}, {end}]'
        )

    bench_df = bench_df.reset_index()
    if 'weight' not in bench_df.columns:
        raise ValueError(
            f"本地 benchmark 权重表缺少 'weight' 列, 实际列={list(bench_df.columns)}"
        )

    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df['weight'] = pd.to_numeric(bench_df['weight'], errors='coerce')
    bench_df = bench_df.dropna(subset=['date', 'order_book_id', 'weight'])

    bench_df = (
        bench_df[['date', 'order_book_id', 'weight']]
        .rename(columns={'weight': 'bench_local_weight'})
        .drop_duplicates(subset=['date', 'order_book_id'], keep='last')
        .sort_values(['date', 'order_book_id'])
    )
    return bench_df


def _load_index_membership_long(
    client: StockDataClient,
    index_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """成分宽表 → 长表，仅保留当日为成分股的行（与 generate_index_weights.py 一致）。"""
    wide = client.get_stock_index_comments(order_book_id=index_code, start=start, end=end)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reset_index()
    date_col = 'date' if 'date' in wide.columns else wide.columns[0]
    idx = wide.melt(
        id_vars=date_col,
        value_name='in_index',
        var_name='order_book_id',
    ).rename(columns={date_col: 'date'})
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx[idx['in_index'].notna() & (idx['in_index'] > 0)]
    return idx[['date', 'order_book_id']]


def _panel_from_index_and_vwap(
    client: StockDataClient,
    index_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """成分长表 + 与 vwap 交易日做内连接对齐（与 generate_index_weights.py 一致）。"""
    idx = _load_index_membership_long(client, index_code, start, end)

    obids = idx['order_book_id'].dropna().unique().tolist()
    if not obids:
        raise ValueError(
            f'指数 {index_code} 在区间 [{start}, {end}] 内无成分股，请检查 index_code 与日期'
        )

    vwap = (
        client.get_stock_post_vwap_1d_data(order_book_ids=list(obids), start=start, end=end)
              .rename(columns={'close': 'vwap'})['vwap']
              .reset_index()
    )
    vwap['date'] = pd.to_datetime(vwap['date'])

    panel = pd.merge(
        idx[['date', 'order_book_id']],
        vwap[['date', 'order_book_id']],
        on=['date', 'order_book_id'],
        how='inner',
    ).sort_values(['order_book_id', 'date'])
    return panel


def _panel_from_index_and_close(
    client: StockDataClient,
    index_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """成分长表 + 与日频除权报价交易日对齐, 同时保留 ``close`` 价格列。"""
    idx = _load_index_membership_long(client, index_code, start, end)

    obids = idx['order_book_id'].dropna().unique().tolist()
    if not obids:
        raise ValueError(
            f'指数 {index_code} 在区间 [{start}, {end}] 内无成分股, 请检查 index_code 与日期'
        )

    close = _load_close_panel(order_book_ids=obids, start=start, end=end)
    panel = pd.merge(
        idx[['date', 'order_book_id']],
        close[['date', 'order_book_id', 'close']],
        on=['date', 'order_book_id'],
        how='inner',
    ).sort_values(['order_book_id', 'date'])
    return panel


def _compute_adjusted_mkt_cap(panel: pd.DataFrame) -> pd.DataFrame:
    """
    在 ``panel`` (含 date / order_book_id / close) 基础上, 按指数公司公布的
    “自由流通分级靠档” 口径计算每只股票的 **调整 (流通) 市值**:

        free_float_ratio = free_circulation / total
        weight_ratio     = LayeredTable(free_float_ratio)
        adj_shares       = total × weight_ratio
        mkt_cap_proxy    = adj_shares × close

    返回 DataFrame 在原列基础上追加:
        ['total', 'free_circulation', 'free_float_ratio',
         'weight_ratio', 'adj_shares', 'mkt_cap_proxy']
    """
    if panel.empty:
        raise ValueError('_compute_adjusted_mkt_cap: 输入 panel 为空')
    if 'close' not in panel.columns:
        raise ValueError(
            "_compute_adjusted_mkt_cap: 输入 panel 缺少 'close' 列, "
            '请先用 _panel_from_index_and_close 构建面板'
        )

    obids = panel['order_book_id'].dropna().unique().tolist()
    start = pd.to_datetime(panel['date'].min()).strftime('%Y-%m-%d')
    end = pd.to_datetime(panel['date'].max()).strftime('%Y-%m-%d')

    shares = _load_shares_panel(order_book_ids=obids, start=start, end=end)
    merged = panel.merge(
        shares[['date', 'order_book_id', 'total', 'free_circulation']],
        on=['date', 'order_book_id'],
        how='left',
    )
    merged = merged.dropna(subset=['total', 'free_circulation'])
    merged = merged[(merged['total'] > 0) & (merged['free_circulation'] >= 0)]

    merged['free_float_ratio'] = (
        merged['free_circulation'].astype(float) / merged['total'].astype(float)
    )
    merged['free_float_pct'] = merged['free_float_ratio'] * 100.0
    merged['weight_ratio'] = merged['free_float_pct'].apply(_weight_ratio_from_free_float_pct)
    merged = merged.dropna(subset=['weight_ratio'])

    merged['adj_shares'] = merged['total'].astype(float) * merged['weight_ratio'].astype(float)
    merged['mkt_cap_proxy'] = merged['adj_shares'] * merged['close'].astype(float)

    merged = merged.dropna(subset=['mkt_cap_proxy'])
    merged = merged[merged['mkt_cap_proxy'] > 0]
    return merged


def _compute_mkt_cap_proxy(
    panel: pd.DataFrame,
    factor_name: str = 'lcap',
    linear_lcap: bool = False,
) -> pd.DataFrame:
    """[回退口径] 在面板上调 add_factor 取得 lcap 因子并换算成市值代理。"""
    factors = myrqfactor.add_factor(panel, factor_name=factor_name)
    if factor_name not in factors.columns:
        raise ValueError(
            f'add_factor 后未找到列 {factor_name!r}, 实际列: {list(factors.columns)}'
        )

    factors = factors.copy()
    col = factors[factor_name].astype(float)
    if linear_lcap:
        factors['mkt_cap_proxy'] = col.clip(lower=0)
    else:
        factors['mkt_cap_proxy'] = np.exp(col)

    factors = factors.dropna(subset=['mkt_cap_proxy'])
    factors = factors[factors['mkt_cap_proxy'] > 0]
    return factors


def _compute_self_weight_panel(
    client: StockDataClient,
    index_code: str,
    start: str,
    end: str,
    factor_name: str = 'lcap',
    linear_lcap: bool = False,
    cap_method: str = 'adjusted',
    benchmark: str = None,
) -> pd.DataFrame:
    """
    计算 index_code 成分股权重, 输出长表 [date, order_book_id, weight]。

    cap_method:
      - 'adjusted' (默认):
          * 若 benchmark 为 None 或 benchmark == index_code:
              直接读本地数据库 index_code 的官方权重 (失败时回退到 adj_cap 自身归一化);
          * 若 benchmark != index_code (典型场景: index=zz1800, benchmark=000300):
              对 (date, order_book_id) 做混合口径:
                - 该股票当日属于 benchmark 成分股 → weight = 本地存储官方权重
                - 该股票当日不属于 benchmark      → weight = adj_cap_i / Σ_{k∈benchmark} adj_cap_k
              与 generate_index_weights.py / compute_index_weight_limits 的 adjusted
              口径完全对齐。
      - 'lcap': 旧版口径, weight_i = exp(lcap_i) / Σ exp(lcap_k) (仅对 index_code 自身归一)。
    """
    if cap_method not in ('adjusted', 'lcap'):
        raise ValueError(f"未知 cap_method={cap_method!r}, 仅支持 'adjusted' / 'lcap'")

    use_mixed = (
        cap_method == 'adjusted'
        and benchmark is not None
        and benchmark != index_code
    )

    # ------------------------------------------------------------------
    # 路径 1: 直接读 index_code 本地官方权重 (index_code 自身, 或 benchmark==index_code)
    # ------------------------------------------------------------------
    if cap_method == 'adjusted' and not use_mixed:
        try:
            local = _load_local_benchmark_weights(client, index_code, start, end)
            if not local.empty:
                out = (
                    local
                    .rename(columns={'bench_local_weight': 'weight'})
                    [['date', 'order_book_id', 'weight']]
                    .copy()
                )
                out['weight'] = pd.to_numeric(out['weight'], errors='coerce')
                out = out.dropna(subset=['date', 'order_book_id', 'weight'])
                out = out[out['weight'] > 0]
                out = out.drop_duplicates(subset=['date', 'order_book_id'], keep='last')
                out = out.sort_values(['date', 'order_book_id']).reset_index(drop=True)
                if not out.empty:
                    daily_sum = out.groupby('date')['weight'].sum()
                    _log(
                        f'本地存储指数权重加载完成: index={index_code}, rows={len(out)}, '
                        f'days={out["date"].nunique()}, '
                        f'daily weight sum min={daily_sum.min():.4f} '
                        f'mean={daily_sum.mean():.4f} max={daily_sum.max():.4f}'
                    )
                    return out
        except Exception as exc:
            _log(
                f'本地存储指数权重加载失败({index_code}): {exc!r}, '
                f'回退到分级靠档调整市值占比口径'
            )

        # fallback: 分级靠档调整市值在 index_code 自身归一化
        panel = _panel_from_index_and_close(client, index_code, start, end)
        factors = _compute_adjusted_mkt_cap(panel)
        daily_total = factors.groupby('date')['mkt_cap_proxy'].transform('sum')
        factors = factors.assign(_total=daily_total)
        factors = factors[factors['_total'] > 0].copy()
        factors['weight'] = (
            factors['mkt_cap_proxy'].astype(float) / factors['_total'].astype(float)
        )
        return _finalize_weight_panel(
            factors[['date', 'order_book_id', 'weight']],
            log_prefix=(
                f'本地市值加权成分股权重(自身归一/fallback): index={index_code}, '
                f'cap_method=adjusted'
            ),
        )

    # ------------------------------------------------------------------
    # 路径 2: 混合口径 (sample=index_code, benchmark=benchmark)
    # ------------------------------------------------------------------
    if use_mixed:
        # 2a) 样本调整市值
        sample_panel = _panel_from_index_and_close(client, index_code, start, end)
        sample_factors = _compute_adjusted_mkt_cap(sample_panel)

        # 2b) 基准总调整市值 (作 adj_cap 占比口径的分母)
        if index_code == benchmark:
            benchmark_totals = (
                sample_factors.groupby('date')['mkt_cap_proxy']
                .sum().rename('bench_total')
            )
        else:
            bench_panel = _panel_from_index_and_close(client, benchmark, start, end)
            bench_factors = _compute_adjusted_mkt_cap(bench_panel)
            benchmark_totals = (
                bench_factors.groupby('date')['mkt_cap_proxy']
                .sum().rename('bench_total')
            )

        merged = sample_factors.merge(
            benchmark_totals.reset_index(), on='date', how='inner',
        )
        merged['weight_calc'] = (
            merged['mkt_cap_proxy'].astype(float) / merged['bench_total'].astype(float)
        )

        # 2c) 用本地基准权重覆盖在 benchmark 成分内的股票
        try:
            bench_local = _load_local_benchmark_weights(
                client=client, benchmark=benchmark, start=start, end=end,
            )
        except Exception as exc:
            _log(
                f'本地 benchmark 权重加载失败({benchmark}): {exc!r}, '
                f'全部样本股将退化到调整市值占比口径'
            )
            bench_local = pd.DataFrame(
                columns=['date', 'order_book_id', 'bench_local_weight']
            )

        merged = merged.merge(
            bench_local, on=['date', 'order_book_id'], how='left',
        )
        merged['in_benchmark'] = merged['bench_local_weight'].notna()
        merged['weight'] = np.where(
            merged['in_benchmark'],
            merged['bench_local_weight'],
            merged['weight_calc'],
        )

        n_in = int(merged['in_benchmark'].sum())
        n_out = int(len(merged) - n_in)
        _log(
            f'样本指数权重(混合口径): sample={index_code}, benchmark={benchmark}, '
            f'本地基准成分内 {n_in} 行 → 本地权重; 非基准成分 {n_out} 行 → adj_cap 占比'
        )

        return _finalize_weight_panel(
            merged[['date', 'order_book_id', 'weight']],
            log_prefix=(
                f'样本指数权重完成(混合口径): sample={index_code}, benchmark={benchmark}'
            ),
        )

    # ------------------------------------------------------------------
    # 路径 3: cap_method='lcap' 旧口径
    # ------------------------------------------------------------------
    panel = _panel_from_index_and_vwap(client, index_code, start, end)
    factors = _compute_mkt_cap_proxy(panel, factor_name=factor_name, linear_lcap=linear_lcap)
    daily_total = factors.groupby('date')['mkt_cap_proxy'].transform('sum')
    factors = factors.assign(_total=daily_total)
    factors = factors[factors['_total'] > 0].copy()
    factors['weight'] = (
        factors['mkt_cap_proxy'].astype(float) / factors['_total'].astype(float)
    )
    return _finalize_weight_panel(
        factors[['date', 'order_book_id', 'weight']],
        log_prefix=(
            f'本地市值加权成分股权重(自身归一): index={index_code}, cap_method=lcap'
        ),
    )


def _finalize_weight_panel(df: pd.DataFrame, log_prefix: str = '') -> pd.DataFrame:
    """规范化输出: 强类型 + 去重 + 排序 + 日志。"""
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out['weight'] = pd.to_numeric(out['weight'], errors='coerce')
    out = out.dropna(subset=['date', 'order_book_id', 'weight'])
    out = out[out['weight'] > 0]
    out = out.drop_duplicates(subset=['date', 'order_book_id'], keep='last')
    out = out.sort_values(['date', 'order_book_id']).reset_index(drop=True)

    n_days = out['date'].nunique()
    if n_days > 0 and log_prefix:
        daily_sum = out.groupby('date')['weight'].sum()
        _log(
            f'{log_prefix}, rows={len(out)}, days={n_days}, '
            f'daily weight sum min={daily_sum.min():.4f} '
            f'mean={daily_sum.mean():.4f} max={daily_sum.max():.4f}'
        )
    return out


def compute_index_weight_limits(
    sample_range: str,
    bench_mark: str,
    start: str,
    end: str,
    data_path: str,
    stock_weight_multiplier: float = 1.1,
    stock_weight_fallback: float = 0.0125,
    stock_weight_cap: float = None,
    factor_name: str = 'lcap',
    linear_lcap: bool = False,
    client: StockDataClient = None,
    cap_method: str = 'adjusted',
) -> pd.DataFrame:
    """
    计算 sample_range 中每只股票相对 bench_mark 的权重, 并构造动态单票上限。

    cap_method:
      - 'adjusted' (默认): 与 PBackTest-dev1-V0915-opt/generate_index_weights.py 一致的混合口径
            * (date, order_book_id) 在 bench_mark 当日成分股内 → 直接使用本地存储官方权重
            * 否则 → adj_cap_i / Σ_{k∈bench_mark} adj_cap_k
              其中 adj_cap = total × LayeredTable(free_circulation/total) × close
              (close 为 rqdatac.get_price 的除权报价)
      - 'lcap'    : 旧版 lcap/exp(lcap) 因子口径, 仅作向后兼容

    单票上限统一:
        max_weight = np.maximum(weight × stock_weight_multiplier, stock_weight_fallback)
        若 stock_weight_cap 非 None，则再限制 max_weight <= stock_weight_cap

    Parameters
    ----------
    sample_range : 样本指数代码 (分子, 例如 'zz1800' / '000906.XSHG')
    bench_mark   : 基准指数代码 (分母, 例如 '000300.XSHG')
    start, end   : 起止日期; 实盘单日场景可传 start=end=pre_date
    data_path    : 行情数据根目录
    stock_weight_multiplier / stock_weight_fallback : 动态单票上限系数
    stock_weight_cap : 可选单票权重绝对上限，None 表示不启用
    factor_name / linear_lcap : 仅在 cap_method='lcap' 时生效
    client       : 可选, 若已有 StockDataClient 实例可复用
    cap_method   : 'adjusted' (默认) / 'lcap'

    Returns
    -------
    DataFrame : MultiIndex(date, order_book_id), 列 ['weight', 'max_weight']
    """
    if client is None:
        print(f'data_path:{data_path}')
        client = StockDataClient(data_path=data_path)

    _log(
        f'开始计算个股基准权重: sample={sample_range}, benchmark={bench_mark}, '
        f'start={start}, end={end}, cap_method={cap_method}'
    )

    if cap_method == 'adjusted':
        sample_panel = _panel_from_index_and_close(client, sample_range, start, end)
        sample_factors = _compute_adjusted_mkt_cap(sample_panel)

        if bench_mark == sample_range:
            _log('sample_range == bench_mark, 复用样本调整市值合计作分母')
            benchmark_totals = (
                sample_factors.groupby('date')['mkt_cap_proxy'].sum().rename('bench_total')
            )
        else:
            bench_panel = _panel_from_index_and_close(client, bench_mark, start, end)
            bench_factors = _compute_adjusted_mkt_cap(bench_panel)
            benchmark_totals = (
                bench_factors.groupby('date')['mkt_cap_proxy'].sum().rename('bench_total')
            )
    elif cap_method == 'lcap':
        sample_panel = _panel_from_index_and_vwap(client, sample_range, start, end)
        sample_factors = _compute_mkt_cap_proxy(sample_panel, factor_name, linear_lcap)
        if bench_mark == sample_range:
            _log('sample_range == bench_mark, 复用样本市值合计作分母 (lcap)')
            benchmark_totals = (
                sample_factors.groupby('date')['mkt_cap_proxy'].sum().rename('bench_total')
            )
        else:
            bench_panel = _panel_from_index_and_vwap(client, bench_mark, start, end)
            bench_factors = _compute_mkt_cap_proxy(bench_panel, factor_name, linear_lcap)
            benchmark_totals = (
                bench_factors.groupby('date')['mkt_cap_proxy'].sum().rename('bench_total')
            )
    else:
        raise ValueError(f"未知 cap_method={cap_method!r}, 仅支持 'adjusted' / 'lcap'")

    benchmark_totals = benchmark_totals[
        benchmark_totals.apply(lambda x: np.isfinite(x) and x > 0)
    ]
    if benchmark_totals.empty:
        raise ValueError(
            f'基准指数 {bench_mark} 在区间 [{start}, {end}] 内未取到有效调整市值'
        )

    merged = sample_factors.merge(
        benchmark_totals.reset_index(), on='date', how='inner'
    )
    merged['weight_calc'] = (
        merged['mkt_cap_proxy'].astype(float) / merged['bench_total'].astype(float)
    )

    # cap_method='adjusted' 走混合口径: benchmark 内股票优先用本地权重
    if cap_method == 'adjusted':
        try:
            bench_local = _load_local_benchmark_weights(
                client=client, benchmark=bench_mark, start=start, end=end,
            )
        except Exception as exc:
            _log(
                f'本地 benchmark 权重加载失败({bench_mark}): {exc!r}, '
                f'全部成分股将退化到调整市值占比口径'
            )
            bench_local = pd.DataFrame(
                columns=['date', 'order_book_id', 'bench_local_weight']
            )

        merged = merged.merge(
            bench_local, on=['date', 'order_book_id'], how='left'
        )
        merged['in_benchmark'] = merged['bench_local_weight'].notna()
        merged['weight'] = np.where(
            merged['in_benchmark'],
            merged['bench_local_weight'],
            merged['weight_calc'],
        )
        n_in = int(merged['in_benchmark'].sum())
        n_out = int(len(merged) - n_in)
        _log(
            f'权重来源: 本地基准成分股 {n_in} 行 -> 本地存储权重; '
            f'非基准成分股 {n_out} 行 -> 调整市值占比'
        )
    else:
        merged['weight'] = merged['weight_calc']

    merged = merged[(merged['weight'] > 0) & np.isfinite(merged['weight'])]
    if merged.empty:
        raise ValueError(
            f'未能生成任何个股权重: sample={sample_range}, benchmark={bench_mark}, '
            f'start={start}, end={end}'
        )

    merged['max_weight'] = np.maximum(
        merged['weight'].astype(float) * float(stock_weight_multiplier),
        float(stock_weight_fallback),
    )
    if stock_weight_cap is not None:
        merged['max_weight'] = np.minimum(
            merged['max_weight'],
            float(stock_weight_cap),
        )

    result = merged[['date', 'order_book_id', 'weight', 'max_weight']].copy()
    result['date'] = pd.to_datetime(result['date'])
    result = result.drop_duplicates(subset=['date', 'order_book_id'], keep='last')
    result = result[result['max_weight'] > 0]
    result = result.set_index(['date', 'order_book_id']).sort_index()

    n_days = result.index.get_level_values('date').nunique()
    daily_sum = result.groupby(level='date')['weight'].sum()
    _log(
        f'个股基准权重完成: rows={len(result)}, days={n_days}, '
        f'daily weight sum min={daily_sum.min():.4f} mean={daily_sum.mean():.4f} '
        f'max={daily_sum.max():.4f}, '
        f'max_weight=np.maximum(weight*{stock_weight_multiplier:.3f}, {stock_weight_fallback:.4f})'
        f"{'' if stock_weight_cap is None else f' capped_by={float(stock_weight_cap):.4f}'}"
    )
    return result


def main(SAMPLE_RANGE, BENCHMARK, START, DATA_PATH, WINDOW):
    """
    入口接口保持不变（被 tranform_target.py 调用）：
        stock_factors, index_exposures = factor_explosure.main(
            sample_range, bench_mark, factor_start_date, database_path, factor_rolling_window
        )

    返回:
        stock_factors:  MultiIndex(date, order_book_id) 的 DataFrame，
                        列含 lcap_exposure / srmi_exposure / beta_exposure / liquidity_exposure
                        （视 myrqfactor 是否能成功计算 illiquidity 而定）
        index_exposures: 含 'date' 列的 DataFrame，列与 stock_factors 中 *_exposure 一一对应
    """

    client = StockDataClient(data_path=DATA_PATH)

    # 1) 样本指数（默认 000906.XSHG / zz1800）成分股权重，用于构造横截面候选集合。
    #    走混合口径: sample 内属于 BENCHMARK 成分股的那部分股票直接取本地官方权重,
    #    其余股票走 adj_cap_i / Σ_{benchmark} adj_cap_k 调整市值占比。
    _log(f'开始加载样本指数权重: sample={SAMPLE_RANGE}, benchmark={BENCHMARK} (混合口径)')
    sample_weights = load_index_weights_panel(
        client=client,
        order_book_id=SAMPLE_RANGE,
        start=START,
        benchmark=BENCHMARK,
    )
    # print(f'sample_weights:{sample_weights}')
    # sample_weights.to_csv('/home/zhanggh/TransformTargetData/tmp/20260415/sample_weights.csv')
    sample_ids = sample_weights['order_book_id'].dropna().unique().tolist()
    _log(f'样本指数加载完成: rows={len(sample_weights)}, stocks={len(sample_ids)}')

    # 2) 基准指数（默认 000300.XSHG）成分股权重，用于聚合得到指数级别的暴露;
    #    benchmark 自身权重 → 等价于全部走本地官方权重。
    _log(f'开始加载基准指数权重: benchmark={BENCHMARK}')
    benchmark_weights = load_index_weights_panel(
        client=client, order_book_id=BENCHMARK, start=START,
    )
    # print(f'benchmark_weights:{benchmark_weights}')
    # benchmark_weights.to_csv('/home/zhanggh/TransformTargetData/tmp/20260415/benchmark_weights.csv')
    _log(f'基准指数加载完成: rows={len(benchmark_weights)}')

    sample_panel = sample_weights[['date', 'order_book_id', 'weight']].drop_duplicates().copy()
    sample_panel = sample_panel.sort_values(['order_book_id', 'date']).dropna()

    # 3) 个股 vwap 与对数收益（与 backtest 一致：用 vwap.pct_change 作为滚动 beta 的 y）
    _log(f'开始加载样本股票 VWAP: stocks={len(sample_ids)}')
    vwap_panel = (
        client.get_stock_post_vwap_1d_data(start=START, order_book_ids=list(sample_ids))
              .rename(columns={'close': 'vwap'})['vwap']
              .reset_index()
    )
    vwap_panel['date'] = pd.to_datetime(vwap_panel['date'])
    vwap_panel = vwap_panel.sort_values(['order_book_id', 'date'])
    vwap_panel['return'] = vwap_panel.groupby('order_book_id')['vwap'].pct_change()
    factor_input = vwap_panel[['date', 'order_book_id', 'return']].copy()
    end_date = factor_input['date'].max()
    _log(f'VWAP 收益面板构建完成: rows={len(factor_input)}, end_date={end_date}')

    # 4) 手工因子（ddnbt240 / uupbt240）：基于 close 收益与基准收益做条件 beta
    manual_factor_panel = build_manual_factor_panel(
        base_df=factor_input[['date', 'order_book_id']],
        client=client,
        order_book_ids=sample_ids,
        benchmark_code=BENCHMARK,
        start=START,
        end=end_date,
        window=int(WINDOW),
    )

    # 5) 通过 myrqfactor 计算其余原始因子（lcap / srmi / illiquidity 等）
    factor_names = sorted({
        factor_name
        for components in BARRA_PROXY_FACTOR_SPECS.values()
        for factor_name, _ in components
    })
    _log(f'待计算原始因子列表: {factor_names}')
    factor_panel, used_factor_names, failed_factors = add_factor_columns(
        factor_input, factor_names, manual_factor_panel=manual_factor_panel
    )
    _log(f'原始因子全部完成: {used_factor_names}')
    if failed_factors:
        _log(f'原始因子失败汇总: {failed_factors}')
    else:
        _log('原始因子失败汇总: 无')

    # 6) 滚动 beta 合成各 *_exposure 列
    barra_exposures = compute_rolling_beta_exposures(
        factor_panel,
        BARRA_PROXY_FACTOR_SPECS,
        window=int(WINDOW),
        ycol='return',
    )

    # 7) nl_size：仅在样本截面上做横截面残差
    _log('开始计算 nl_size')
    nl_input = sample_panel[['date', 'order_book_id']].drop_duplicates().merge(
        factor_panel[['date', 'order_book_id', 'lcap']],
        on=['date', 'order_book_id'],
        how='left',
    )
    nl_size = compute_nl_size(nl_input[['date', 'order_book_id', 'lcap']].dropna())
    _log(f'nl_size 计算完成: 非空行数={int(nl_size["nl_size"].notna().sum())}')

    stock_factors = barra_exposures.merge(nl_size, on=['date', 'order_book_id'], how='left')
    for canonical_col, source_col in LEGACY_ALIAS_MAP.items():
        if source_col in stock_factors.columns:
            stock_factors[canonical_col] = stock_factors[source_col]
    _log(f'个股暴露表构建完成: rows={len(stock_factors)}, cols={list(stock_factors.columns)}')

    ordered_cols = [c for c in STOCK_OUTPUT_COLS if c in stock_factors.columns]
    stock_factors = (
        stock_factors[ordered_cols]
        .sort_values(['date', 'order_book_id'])
        .reset_index(drop=True)
    )

    # 8) 指数级风格暴露：按基准成分权重做加权平均
    merge_cols = [c for c in stock_factors.columns if c not in ['date', 'order_book_id']]
    benchmark_merge = benchmark_weights.merge(stock_factors, on=['date', 'order_book_id'], how='inner')
    _log(f'基准合并完成: rows={len(benchmark_merge)}, merge_cols={merge_cols}')
    index_exposures = weighted_exposure_by_date(benchmark_merge, merge_cols)
    _log(f'指数暴露表构建完成: rows={len(index_exposures)}, cols={list(index_exposures.columns)}')

    # 与原 trade 接口保持一致：stock_factors 设 MultiIndex，index_exposures 保持 'date' 列
    stock_factors_indexed = (
        stock_factors.set_index(['date', 'order_book_id']).sort_index()
    )
    return stock_factors_indexed, index_exposures


if __name__ == "__main__":
    # 默认与 yaml 中保持一致：选股域=000906；指数=000300
    SAMPLE_RANGE = "000906.XSHG"
    BENCHMARK = "000300.XSHG"
    START = "2019-01-01"
    DATA_PATH = "/home/samba/Market/"
    WINDOW = 252

    # 输出路径（两份）
    STOCK_OUTPUT = "/home/zhanggh/TransformTargetData/factors/906_exposures.parquet"
    INDEX_OUTPUT = "/home/zhanggh/TransformTargetData/factors/300_index_exposures.parquet"

    stock_factors, index_exposures = main(SAMPLE_RANGE, BENCHMARK, START, DATA_PATH, WINDOW)

    os.makedirs(os.path.dirname(STOCK_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_OUTPUT), exist_ok=True)
    stock_factors.reset_index().to_parquet(STOCK_OUTPUT)
    index_exposures.to_parquet(INDEX_OUTPUT)
    _log(f'Saved stock exposures to: {STOCK_OUTPUT}')
    _log(f'Saved index daily exposures to: {INDEX_OUTPUT}')
