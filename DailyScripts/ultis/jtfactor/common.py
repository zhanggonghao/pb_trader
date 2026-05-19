import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

# 全局忽略特定的RuntimeWarning
warnings.filterwarnings("ignore")


def get_newey_west_adjust_std(seq, llag_ratio=0.25):
    """计算序列的NW调整标准差

    :return: New-West Adjust Std
    :return type: float
    """
    # 去掉NaN值
    seq = [i for i in seq if not np.isnan(i)]
    _llag = int(len(seq) ** llag_ratio)
    _nw_cov = sm.stats.sandwich_covariance.cov_hac(
        sm.OLS(seq, np.ones(len(seq))).fit(),
        nlags=_llag
    )
    _nw_std_final = np.sqrt(_nw_cov[0, 0])  # 对角线元素是调整后的方差估计
    return _nw_std_final


def mean(x):
    return np.nanmean(x)


def skewness(x):
    n = len(x)
    mean_x = np.mean(x)
    m3 = np.sum((x - mean_x)**3) / n
    s3 = np.std(x, ddof=0)**3  # 样本标准差的三次方
    skew = m3 / s3
    return skew


def kurtosis(x):
    n = len(x)
    mean_x = np.mean(x)
    m4 = np.sum((x - mean_x)**4) / n
    s4 = np.std(x, ddof=0)**4  # 样本标准差的四次方
    kurt = m4 / s4 - 3
    return kurt


def std(x):
    return np.nanstd(x)


def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def multi_vars_regression(y, X, add_const=True):
    """多元回归算子：一对多回归

    """
    if add_const:
        X = np.column_stack([X, np.ones_like(y)])
    # β = (X'X)^(-1)X'Y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    residual = y - X @ beta
    # 当且仅当存在常数项时，最后一个系数对应的值才是alpha，否则默认NaN值
    alpha = np.NaN
    if add_const:
        alpha = beta[-1]
    return beta, residual, alpha


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def roll_window(arr, window):
    """
    滚动提取数组的函数。

    :param arr: 待滚动提取的数组，可以是一维或二维。
    :param window: 滚动的窗口大小。
    :return: 滚动提取后的高维数组。
    """
    # if isinstance(arr, pd.Series):
    #     arr = arr.values
    # 确定新数组的形状
    # new_shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    # if arr.shape[0] - window + 1 < 0:
    #     new_shape = (1, window) + arr.shape[1:]
    # else:
    new_shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    # print(new_shape)
    # 确定新数组的步长
    new_strides = (arr.strides[0],) + arr.strides

    # 使用as_strided创建新视图
    # try:
    rolled_arr = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    # except ValueError:
    #     print(new_shape, arr.shape, window)
    #     raise ValueError("break")
    return rolled_arr


def regression(y, X, add_const=True):
    """回归函数"""
    if add_const:
        X = np.column_stack([X, np.ones_like(y)])
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
    except np.linalg.LinAlgError:
        beta = [np.NaN] * X.shape[1]
    return beta


def ts_regression(y, x, window, add_const=True):
    """滚动时序回归

    - 滚动结果如何计算residual：使用np.einsum
      举例：矩阵Y的形状为(L,N)，矩阵B的形状为(L,K)，矩阵X的形状为(L,N,K)，则
           滚动残差矩阵R=np.einsum('ij,ikj->ik', B, X)，得到R的形状为(L,N)

    :param y: 因变量，(N,)形状，N为样本个数
    :param x: 自变量，(N,K)形状，N为样本个数，K为变量个数
    :param window: 滚动窗口，默认window<=N
    :param add_const: 是否添加常数项，默认添加且添加至最后一列
    """
    r_y = roll_window(y, window)
    if add_const:
        const = np.ones_like(y)
        x = np.column_stack([x, const])
    r_X = roll_window(x, window)

    betas = []
    # residuals = []
    for i in range(len(r_y)):
        y = r_y[i, :]
        x = r_X[i, :, :]
        # beta = np.linalg.inv(x.T @ x) @ x.T @ y
        beta = regression(y, x, False)
        # residual = y - x @ beta
        betas.append(beta)
    return betas


# def rolling_multi_vars_regression(y, X, window, add_const=True):
#     """时序移动窗口多元回归算子
#
#     :param y: 一维时序值(n,)
#     :param X: K维时序值，默认输入方式为(N*K)，其中每一行是一个观测样本，每一列是一个变量
#     """
#
#     r_y = rolling_window(y, window)
#     list_r_x = []
#     for x in X:
#         list_r_x.append(rolling_window(x, window))
#     if add_const:
#         const = np.ones_like(y)
#         list_r_x.append(rolling_window(const, window))
#     r_X = np.array(list_r_x)
#
#     betas = []
#     residuals = []
#     for i in range(len(r_y)):
#         y = r_y[i, :].T
#         x = r_X[:, i, :].T
#         beta = np.linalg.inv(x.T @ x) @ x.T @ y
#         residual = y - x @ beta
#         betas.append(beta)
#         residuals.append(residual)
#         # X = np.hstack([np.ones((window, 1)), x_window])
#         # 使用最小二乘法计算回归系数
#         # β = (X'X)^(-1)X'y
#         # beta = np.linalg.inv(X.T @ X) @ X.T @ y_window
#         # coefficients.append(beta)
#     return np.array(residuals)[:, -1]

def rolling_zscore(series, window):
    # r = rolling_window(series.values, window)
    r = roll_window(series.values, window)
    _mean = np.nanmean(r, axis=1)[:, np.newaxis]
    _std = np.nanstd(r, axis=1)[:, np.newaxis]
    return ((r - _mean) / _std)[:, -1]



def jt_TWAP(df, *factor_columns):
    """时序加权因子

    时序加权=所有时刻的均值
    """
    factor_columns = list(factor_columns)
    return df[factor_columns].mean(numeric_only=True).squeeze().item()


def jt_TS_CORR(df, *columns, method="spearman"):
    """时序相关系数因子计算

    :param df: 待计算数据，以datetime或date索引，含多列相关数据。
    :type df: pd.DataFrame
    :param columns: 待计算相关系数的列表，示例：'close', 'open', 'high'
    :type columns: list[str]
    :param method: 相关系数计算方法，默认：spearman（秩相关系数），其余包括：'pearson', 'kendall'

    :return: 计算的相关系数结果。若有3列及以上输入，则返回多个相关系数组成的pd.Series；若输入两列，则返回单一数值
    :return type: pd.Series, float
    """
    # _len = len(columns)
    _cols = list(columns)
    _corr = df[_cols].corr(method=method).iloc[0][1:]
    return _corr.squeeze().item()


def jt_TS_COV(df, *columns):
    _cols = list(columns)
    _cov = df[_cols].cov().iloc[0][1:]
    return _cov.squeeze().item()


def jt_TS_STD(df, *columns):
    _cols = list(columns)
    _std = df[_cols].std()
    return _std.squeeze().item()


def jt_TS_MIN(df, *columns):
    _cols = list(columns)
    _min = df[_cols].min()
    return _min.squeeze().item()


def jt_TS_MAX(df, *columns):
    _cols = list(columns)
    _max = df[_cols].max()
    return _max.squeeze().item()


def jt_TS_SUM(df, *columns):
    _cols = list(columns)
    _sum = df[_cols].sum()
    return _sum.squeeze().item()


def jt_TS_ZSCORE(df, *columns):
    _cols = list(columns)
    _std = df[_cols].std()
    _mean = df[_cols].mean()
    _zscore = df[_cols].div(_std).iloc[-1]  # 这里取最后一行数据（即T0时刻数值）
    # 如果发生停牌，可能出现inf值（当天std==0）
    return _zscore.squeeze().item()


def jt_TS_COV_numpy(df, *columns):
    _cols = list(columns)
    cov_matrix = np.cov(df[_cols].values, rowvar=False)
    return cov_matrix[0, 1:]


def jt_TS_CORR_numpy(df, *columns):
    _cols = list(columns)
    df_values = df[_cols].values
    corr_matrix = np.corrcoef(df_values, rowvar=False)
    return corr_matrix[0, 1:]
