"""
配置模块 - 管理所有配置参数

本文件集中管理报告生成所需的所有配置参数，包括：
- 路径配置
- 产品信息
- 报告周期
- 米筐API配置
- 计算参数
- 输出配置
"""
import os

# ==============================================================================
# 基础路径配置
# ==============================================================================
input_path = r'E:\code\generate_split_system\data' # 交易机挂载目录
BASE_DIR = r"E:\code\product_report"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DATA_DIR = os.path.join(BASE_DIR, "temp_data")
INTERMEDIATE_DATA_DIR = os.path.join(OUTPUT_DIR, "intermediate_data")

# 数据源子目录
NET_EMAIL_DIR = os.path.join(input_path, "raw", "net_email")
STK_FUT_DIR = os.path.join(input_path, "out", "stk_fut")
POS_DIR = os.path.join(input_path, "standarddata", "pos")
OUT_IN_DIR = input_path

# ==============================================================================
# 产品信息配置
# ==============================================================================
REPORT_PRODUCT_CODE = "PBHSZX1H"
REPORT_PRODUCT_NAME = "配邦恒升中性1号"

# ==============================================================================
# 报告周期配置
# ==============================================================================
REPORT_START = "20260518"
REPORT_END = "20260522"

# ==============================================================================
# 基准配置
# ==============================================================================
BENCHMARK_CODE = "000300.XSHG"
BENCHMARK_NAME = "沪深300指数"

# ==============================================================================
# 米筐API配置
# ==============================================================================
RQ_USERNAME = "license"
RQ_PASSWORDS = [
    "gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=",
    "jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=",
    ]
RQ_TIMEOUT = 30  # 米筐连接超时时间（秒）

# ==============================================================================
# 计算参数配置
# ==============================================================================
class FactorConfig:
    """
    因子数据配置类
    """
    # 因子文件路径，168服务器挂载目录
    BENCHMARK_FACTOR_PATH = r'Y:\Market\factors\factors_post_1d\adjusted_300_index_exposures.parquet'
    STOCK_FACTOR_PATH = r'Y:\Market\factors\factors_post_1d\adjusted_1800_vs_300exposures.parquet'
    
    # 选择的因子列表
    SELECTED_FACTORS = {
        'lcap_exposure': '市值',
        'liquidity_exposure': '流动性',
        'beta_exposure': 'Beta',
    }


# ==============================================================================
class CalculationParams:
    """
    计算参数配置类
    
    包含所有报告指标的计算参数：
    """
    
    # 期货相关参数
    FUTURE_MARGIN_RATIO = 0.12  # 期货保证金比例（用于计算名义市值）
    
    # 持仓集中度参数
    CONCENTRATION_TOP_PCT = 0.2  # 计算集中度时取前N%的股票（默认20%）
    
    # 股票代码转换规则
    STOCK_CODE_RULES = {
        'SH': ['60', '688'],  # 沪市股票前缀
        'SZ': ['00', '002', '000', '30', '20'],  # 深市股票前缀
    }
    
    # 因子列表
    FACTOR_LIST = [
        'size',              # 规模因子，lcap_exposure
        'non_linear_size',   # 非线性规模因子
        'momentum',          # 动量因子，srmi_exposure，先不算
        'liquidity',         # 流动性因子，liquidity_exposure
        'book_to_price',     # 价值因子（市净率）
        'leverage',          # 杠杆因子
        'growth',            # 成长因子
        'earnings_yield',    # 盈利因子
        'beta',              # Beta因子， beta_exposure
        'residual_volatility',  # 残差波动因子
    ]
    
    # 因子中文名称映射
    FACTOR_CN_NAMES = {
        'size': '规模',
        'non_linear_size': '非线性规模',
        'momentum': '动量',
        'liquidity': '流动性',
        'book_to_price': '价值',
        'leverage': '杠杆',
        'growth': '成长',
        'earnings_yield': '盈利',
        'beta': 'Beta',
        'residual_volatility': '残差波动',
    }

# ==============================================================================
# PDF报告配置
# ==============================================================================
class PDFConfig:
    """
    PDF报告配置类
    
    包含PDF生成的相关配置：
    """
    
    # 报告标题
    REPORT_TITLE = "量化基金产品周期报告"
    
    # 页面设置
    PAGE_WIDTH = 210  # A4宽度（mm）
    PAGE_HEIGHT = 297  # A4高度（mm）
    MARGIN = 20  # 页面边距（mm）
    
    # 字体设置
    FONT_NAME = "SimHei"
    FONT_SIZE_TITLE = 16
    FONT_SIZE_SECTION = 14
    FONT_SIZE_SUBSECTION = 12
    FONT_SIZE_BODY = 10
    FONT_SIZE_SMALL = 8
    
    # 颜色设置
    ACCENT_COLOR = '#1a3a5c'  # 主色调
    POSITIVE_COLOR = '#27ae60'  # 正数颜色（绿色）
    NEGATIVE_COLOR = '#c0392b'  # 负数颜色（红色）
    
    # 图表设置
    CHART_DPI = 150
    CHART_WIDTH = 10
    CHART_HEIGHT = 4
    
    # 模块显示控制
    SHOW_MODULE1 = True  # 一、周期净值信息
    SHOW_MODULE2 = True  # 二、每日收益明细
    SHOW_MODULE3 = False  # 三、多空详情（已禁用）
    SHOW_MODULE4 = True  # 四、每日个股权重Top10
    SHOW_MODULE5 = True  # 五、持仓集中度
    SHOW_MODULE6 = True  # 六、因子风格敞口

# ==============================================================================
# 输出文件命名规则
# ==============================================================================
def get_report_filename(product_code, start_date, end_date):
    """生成报告文件名"""
    return f"{product_code}_{start_date}_{end_date}_报告.pdf"

def get_intermediate_data_filename(module_name):
    """生成中间数据文件名"""
    return f"module{module_name}.csv"

# ==============================================================================
# 确保目录存在
# ==============================================================================
def ensure_directories():
    """确保所有必要目录存在"""
    directories = [
        OUTPUT_DIR,
        TEMP_DATA_DIR,
        INTERMEDIATE_DATA_DIR,
    ]
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# 初始化目录
ensure_directories()