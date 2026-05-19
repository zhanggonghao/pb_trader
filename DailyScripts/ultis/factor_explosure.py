import os
import pandas as pd
import numpy as np

import rqdatac
# rqdatac.init(13601611030,'PB123456789')
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)


from ultis.stock_data_client import StockDataClient
from ultis import myrqfactor
 

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


def compute_rolling_exposures(df: pd.DataFrame, window: int = 252, ycol: str = 'return') -> pd.DataFrame:
    """
    计算 lcap_exposure 与 srmi_exposure：
    对每只股票，beta = cov(x, y) / var(x)，min_periods=window，ddof=0。
    需要列：['date','order_book_id', ycol, 'lcap','srmi']。
    返回：['date','order_book_id','lcap_exposure','srmi_exposure']。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['order_book_id', 'date'])

    out_l = []
    out_s = []
    for oid, g in df.groupby('order_book_id'):
        y = g[ycol].astype(float)

        x_l = g['lcap'].astype(float)
        var_l = x_l.rolling(window, min_periods=window).var(ddof=0)
        cov_l = x_l.rolling(window, min_periods=window).cov(y)
        beta_l = cov_l / var_l
        beta_l[var_l <= 0] = np.nan
        out_l.append(beta_l)

        x_s = g['srmi'].astype(float)
        var_s = x_s.rolling(window, min_periods=window).var(ddof=0)
        cov_s = x_s.rolling(window, min_periods=window).cov(y)
        beta_s = cov_s / var_s
        beta_s[var_s <= 0] = np.nan
        out_s.append(beta_s)

    expo = df[['date','order_book_id']].copy()
    expo['lcap_exposure'] = pd.concat(out_l).sort_index().values
    expo['srmi_exposure'] = pd.concat(out_s).sort_index().values
    return expo


def main(SAMPLE_RANGE, BENCHMARK, START, DATA_PATH, WINDOW):

    client = StockDataClient(data_path=DATA_PATH)

    # 1) 000906 成分
    idx_906 = client.get_stock_index_comments_weights(order_book_id=SAMPLE_RANGE, start=START).reset_index()
    # print('idx_906\n', idx_906)
    idx_906 = idx_906.melt(id_vars='date', value_name='weight', var_name='order_book_id')
    idx_906['date'] = pd.to_datetime(idx_906['date'])
    idx_906['weight'] = idx_906.sort_values(['order_book_id','date']).groupby('order_book_id')['weight'].shift(1)
    obids_906 = idx_906['order_book_id'].dropna().unique().tolist()


    # 2) 个股 vwap 与收益
    vwap_906 = (
        client.get_stock_post_vwap_1d_data(start=START, order_book_ids=list(obids_906))
              .rename(columns={'close': 'vwap'})['vwap']
              .reset_index()
    )
    vwap_906['date'] = pd.to_datetime(vwap_906['date'])
    vwap_906 = vwap_906.sort_values(['order_book_id', 'date'])
    vwap_906['return'] = vwap_906.groupby('order_book_id')['vwap'].pct_change()

    # 3) 计算所需面板
    # 3.1 当日成分面板（仅用于 nl_size 的横截面）
    panel_906 = pd.merge(
        idx_906[['date','order_book_id','weight']],
        vwap_906[['date','order_book_id']],
        on=['date','order_book_id'], how='outer'
    ).sort_values(['order_book_id','date'])
    # 平移权重到下一天
    # print('idx_906\n', idx_906.dropna())
    panel_906['weight'] = panel_906.groupby('order_book_id')['weight'].shift(1).dropna(how='any')
    # print('panel_906\n', panel_906.dropna())
    # print(panel_906.tail(5))

    # 3.2 基于成分面板计算 nl_size 所需的 lcap（仅成分内）
    factors_cs = myrqfactor.add_factor(panel_906, factor_name='lcap').dropna()
    # print('factors_cs\n', factors_cs)
    # print(factors_cs.tail(5))

    # 3.3 滚动暴露：不筛选当日成分，直接在全量 vwap_906 上计算 lcap/srmi
    factors_all = myrqfactor.add_factor(vwap_906, factor_name='lcap')
    factors_all = myrqfactor.add_factor(factors_all, factor_name='srmi')
    # print('factors_all\n', factors_all)

    df_panel = pd.merge(
        vwap_906[['date','order_book_id','return']],
        factors_all[['date','order_book_id','lcap','srmi']],
        on=['date','order_book_id'], how='left'
    ).sort_values(['order_book_id','date'])
    stock_exposures = compute_rolling_exposures(df_panel, window=WINDOW, ycol='return').dropna()

    # 4) nl_size（仅在当日成分截面上）
    nl = compute_nl_size(factors_cs[['date','order_book_id','lcap']].dropna())
    stock_factors = stock_exposures.merge(nl, on=['date','order_book_id'], how='left')
    # print('stock_factors\n', stock_factors)
    
    # print(stock_factors.tail(5))

    # 6) 保存个股文件（含 lcap_exposure, srmi_exposure, nl_size）
    # os.makedirs(os.path.dirname(STOCK_OUTPUT), exist_ok=True)
    # stock_factors.to_parquet(STOCK_OUTPUT)

    # 7) 计算 000300 指数每天的暴露/因子（权重加总）
    idx_300 = client.get_stock_index_comments_weights(order_book_id=BENCHMARK, start=START).reset_index()
    idx_300 = idx_300.melt(id_vars='date', value_name='weight', var_name='order_book_id')
    idx_300['date'] = pd.to_datetime(idx_300['date'])
    idx_300['weight'] = idx_300.sort_values(['order_book_id','date']).groupby('order_book_id')['weight'].shift(1)

    idx_merge = pd.merge(
        idx_300[['date','order_book_id','weight']],
        stock_factors, on=['date','order_book_id'], how='outer'
    )

    # 平移权重到下一天
    idx_merge = idx_merge.sort_values(['order_book_id','date'])
    idx_merge['weight'] = idx_merge.groupby('order_book_id')['weight'].shift(1).dropna()
    # print('idx_merge\n', idx_merge)

    index_lcap = idx_merge.groupby('date').apply(lambda x: (x['lcap_exposure'] * x['weight']).sum()).reset_index()
    index_lcap.columns = ['date', 'lcap_exposure']

    index_srmi = idx_merge.groupby('date').apply(lambda x: (x['srmi_exposure'] * x['weight']).sum()).reset_index()
    index_srmi.columns = ['date', 'srmi_exposure']

    index_nls = idx_merge.groupby('date').apply(lambda x: (x['nl_size'] * x['weight']).sum()).reset_index()
    index_nls.columns = ['date', 'nl_size']

    index_exposures = index_lcap.merge(index_srmi, on='date').merge(index_nls, on='date')
    # print('index_exposures\n', index_exposures)
    
    # print(index_exposures.tail(5))

    # 8) 保存指数文件（含 lcap_exposure, srmi_exposure, nl_size）
    # os.makedirs(os.path.dirname(INDEX_OUTPUT), exist_ok=True)
    # index_exposures.to_parquet(INDEX_OUTPUT)
    # print(stock_factors)
    # print(index_exposures)
    # print(f"Saved stock exposures to: {STOCK_OUTPUT}")
    # print(f"Saved index daily exposures to: {INDEX_OUTPUT}")
    return stock_factors.set_index(['date','order_book_id']).sort_index(), index_exposures


if __name__ == "__main__":
    
    # 选股域=000906；指数=000300
    SAMPLE_RANGE = "000906.XSHG"
    BENCHMARK = "000300.XSHG"
    START = "2019-01-01"
    DATA_PATH = "/home/samba/Market/"
    WINDOW = 252

    # 输出路径（两份）
    STOCK_OUTPUT = "/home/zhanggh/TransformTargetData/factors/906_exposures.parquet"
    INDEX_OUTPUT = "/home/zhanggh/TransformTargetData/factors/300_index_exposures.parquet"
    main(SAMPLE_RANGE, BENCHMARK, START, DATA_PATH, WINDOW)