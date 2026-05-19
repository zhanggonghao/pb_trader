"""
实时行情数据模块
对接米筐 rqdatac，提供：
  1. 股票实时行情
  2. 指数实时行情
  3. 期货实时行情 + 自动获取所有合约
  4. 基差计算
"""
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import rqdatac
import pandas as pd
import numpy as np

from .config import CONFIG

logger = logging.getLogger(__name__)

# 交易时间
MORNING_START = time(9, 30)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(15, 0)


def init() -> bool:
    """初始化米筐数据源"""
    cfg = CONFIG.rqdatac
    try:
        rqdatac.init(
            username=cfg.username,
            password=cfg.password,
            use_pool=True,
            max_pool_size=cfg.max_pool_size
        )
        logger.info("米筐数据源初始化成功")
        return True
    except Exception as e:
        logger.error(f"米筐数据源初始化失败: {e}")
        return False


def is_trading_time() -> bool:
    """判断当前是否在交易时间"""
    now = datetime.now().time()
    weekday = datetime.now().weekday()
    if weekday >= 5:
        return False
    if MORNING_START <= now <= MORNING_END:
        return True
    if AFTERNOON_START <= now <= AFTERNOON_END:
        return True
    return False


def get_today_str() -> str:
    """获取今日日期字符串 YYYYMMDD"""
    return datetime.now().strftime("%Y%m%d")


def get_previous_trading_date() -> str:
    """获取上一个交易日"""
    today = datetime.now().strftime("%Y-%m-%d")
    prev = rqdatac.get_previous_trading_date(today)
    return prev.strftime("%Y-%m-%d")


# ========== 股票实时行情 ==========

def get_stock_prices(codes: List[str]) -> Dict[str, float]:
    """
    获取股票实时最新价
    codes: ["600901.XSHG", "300274.XSHE"]
    returns: {"600901.XSHG": 12.34, ...}
    """
    try:
        df = rqdatac.current_minute(codes, skip_suspended=False)
        if df is None or df.empty:
            return {}
        df = df.reset_index()
        result = {}
        for _, row in df.iterrows():
            code = row.get("order_book_id", "")
            close = row.get("close")
            if code and close is not None and not np.isnan(close):
                result[code] = float(close)
        return result
    except Exception as e:
        logger.warning(f"获取股票行情失败: {e}")
        return {}


# ========== 指数行情 ==========

def get_index_components(index_code: str) -> List[str]:
    """获取指数成分股列表"""
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        df = rqdatac.index_weights(index_code, date=today_str)
        if df is None or df.empty:
            prev = rqdatac.get_previous_trading_date(today_str)
            df = rqdatac.index_weights(index_code, date=prev.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            return df.index.get_level_values('order_book_id').unique().tolist()
        return []
    except Exception as e:
        logger.warning(f"获取指数{index_code}成分股失败: {e}")
        return []


def get_index_price(index_code: str) -> Optional[float]:
    """获取指数实时最新价"""
    try:
        df = rqdatac.current_minute([index_code], skip_suspended=False)
        if df is None or df.empty:
            return None
        return float(df.iloc[-1]["close"])
    except Exception as e:
        logger.warning(f"获取指数{index_code}行情失败: {e}")
        return None


def get_index_prices(index_codes: List[str]) -> Dict[str, float]:
    """批量获取指数实时行情"""
    result = {}
    for code in index_codes:
        price = get_index_price(code)
        if price is not None:
            result[code] = price
    return result


def get_index_prev_close(index_code: str) -> Optional[float]:
    """获取指数昨日收盘价"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        prev = rqdatac.get_previous_trading_date(today)
        df = rqdatac.get_price(
            [index_code], start_date=prev, end_date=prev,
            frequency="1d", fields=["close"]
        )
        if df is not None and not df.empty:
            return float(df.iloc[0]["close"])
        return None
    except Exception as e:
        logger.warning(f"获取指数昨收失败: {e}")
        return None


# ========== 期货行情 ==========

def get_future_contracts(underlying: str) -> List[str]:
    """
    获取某期货品种当前所有可交易合约列表
    underlying: "IF", "IH", "IC", "IM"
    returns: ["IF2506", "IF2509", "IF2512", "IF2603"]
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        contracts = rqdatac.futures.get_contracts(underlying, date=today)
        return contracts if contracts else []
    except Exception as e:
        logger.warning(f"获取{underlying}合约列表失败: {e}")
        return []


def get_dominant_contract(underlying: str) -> Optional[str]:
    """
    获取主力合约
    underlying: "IF"
    returns: "IF2506"
    """
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        series = rqdatac.futures.get_dominant(underlying, start_date=today, end_date=today)
        if series is not None and not series.empty:
            return str(series.iloc[-1])
        return None
    except Exception as e:
        logger.warning(f"获取{underlying}主力合约失败: {e}")
        return None


def get_future_prices(contracts: List[str]) -> Dict[str, dict]:
    """
    获取期货合约实时行情
    returns: {
        "IF2506": {
            "price": 3800.0,
            "prev_close": 3750.0,
            "change_pct": 1.33,
            "volume": 12345,
            "open_interest": 100000
        }
    }
    """
    try:
        df = rqdatac.current_minute(contracts, skip_suspended=False)
        if df is None or df.empty:
            return {}
        df = df.reset_index()
        result = {}
        for _, row in df.iterrows():
            code = row.get("order_book_id", "")
            close = row.get("close")
            if code and close is not None and not np.isnan(close):
                pre_close = row.get("pre_close")
                change_pct = 0.0
                if pre_close and pre_close > 0:
                    change_pct = round((close - pre_close) / pre_close * 100, 2)
                result[code] = {
                    "price": float(close),
                    "prev_close": float(pre_close) if pre_close else 0,
                    "change_pct": change_pct,
                    "volume": int(row.get("volume", 0)),
                    "open_interest": int(row.get("open_interest", 0)),
                }
        return result
    except Exception as e:
        logger.warning(f"获取期货行情失败: {e}")
        return {}


def get_contract_delivery_date(contract: str) -> Optional[datetime]:
    """
    获取期货合约交割日（到期日）
    从合约代码解析: IF2506 -> 2025-06-20 (第三个周五)
    """
    try:
        # 使用米筐获取合约信息
        inst_info = rqdatac.instruments(contract)
        if inst_info is not None and hasattr(inst_info, 'listed_date'):
            # 获取合约到期日
            df = rqdatac.futures.get_contracts(contract[:2], date=datetime.now().strftime("%Y-%m-%d"))
            # 获取合约详情
            from rqdatac import get_instrument_info
            info = get_instrument_info(contract)
            if info and 'maturity_date' in info:
                return pd.Timestamp(info['maturity_date']).to_pydatetime()
        return None
    except Exception as e:
        logger.warning(f"获取合约{contract}到期日失败: {e}")
        return None


def calculate_basis(future_price: float, spot_price: float) -> float:
    """计算基差 = 期货价格 - 现货价格"""
    return future_price - spot_price


def calculate_basis_cost(basis: float, spot_price: float, days_to_expiry: int) -> float:
    """
    计算基差成本（年化）
    成本 = (基差 / 现货价格) * (365 / 剩余天数)
    """
    if spot_price <= 0 or days_to_expiry <= 0:
        return 0.0
    return (basis / spot_price) * (365.0 / days_to_expiry) * 100  # 返回百分比


def get_all_future_data() -> Dict[str, dict]:
    """
    获取所有股指期货品种的完整数据（含基差计算）

    returns: {
        "IF": {
            "spot_index": "000300.XSHG",
            "spot_price": 3900.0,
            "dominant": "IF2506",
            "dominant_price": 3850.0,
            "basis": -50.0,
            "basis_cost_pct": -2.5,
            "contracts": {
                "IF2506": {"price": 3850.0, "basis": -50.0, "basis_cost_pct": -2.5, ...},
                "IF2509": {"price": 3830.0, ...}
            }
        },
        ...
    }
    """
    result = {}

    for fp in CONFIG.futures:
        try:
            # 1. 获取现货指数价格
            spot_price = get_index_price(fp.spot_index)
            if spot_price is None:
                logger.warning(f"无法获取现货指数 {fp.spot_index} 行情")
                continue

            # 2. 获取所有合约
            contracts = get_future_contracts(fp.name)
            if not contracts:
                logger.warning(f"获取 {fp.name} 合约列表为空")
                continue

            # 3. 获取主力合约
            dominant = get_dominant_contract(fp.name)

            # 4. 获取所有合约行情
            prices = get_future_prices(contracts)

            # 5. 计算每个合约的基差
            contract_data = {}
            dominant_data = None

            for contract in contracts:
                if contract not in prices:
                    continue
                p = prices[contract]

                # 计算基差
                basis = calculate_basis(p["price"], spot_price)

                # 估算到期天数（从合约代码解析）
                days_to_expiry = _estimate_days_to_expiry(contract)
                basis_cost = calculate_basis_cost(basis, spot_price, days_to_expiry)

                info = {
                    "contract": contract,
                    "price": p["price"],
                    "prev_close": p["prev_close"],
                    "change_pct": p["change_pct"],
                    "basis": round(basis, 2),
                    "basis_cost_pct": round(basis_cost, 2),
                    "days_to_expiry": days_to_expiry,
                    "volume": p["volume"],
                    "is_dominant": (contract == dominant)
                }
                contract_data[contract] = info

                if contract == dominant:
                    dominant_data = info

            # 构建品种数据
            product_data = {
                "spot_index": fp.spot_index,
                "spot_price": spot_price,
                "dominant": dominant,
                "dominant_price": dominant_data["price"] if dominant_data else 0,
                "dominant_basis": dominant_data["basis"] if dominant_data else 0,
                "dominant_basis_cost": dominant_data["basis_cost_pct"] if dominant_data else 0,
                "contracts": contract_data,
                "contracts_list": sorted(contract_data.keys()),
            }
            result[fp.name] = product_data

        except Exception as e:
            logger.error(f"处理 {fp.name} 数据时出错: {e}")
            continue

    return result


def _estimate_days_to_expiry(contract: str) -> int:
    """
    估算合约到期天数（基于合约代码）
    股指期货到期日为合约月份的第三个周五
    简化为取该月第20日左右
    """
    import re
    from datetime import datetime

    match = re.match(r'([A-Z]+)(\d{4})', contract)
    if not match:
        return 90  # 默认90天

    year = 2000 + int(match.group(2)[:2])
    month = int(match.group(2)[2:])

    # 估算第三个周五
    # 简单处理：取该月第20天
    try:
        expiry = datetime(year, month, min(20, 28))
        delta = (expiry - datetime.now()).days
        return max(delta, 1)
    except:
        return 90
