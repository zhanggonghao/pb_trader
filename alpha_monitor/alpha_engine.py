"""
Alpha 超额收益计算引擎
"""
import csv
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import rqdatac

from .config import CONFIG
from . import market_data as mdata

logger = logging.getLogger(__name__)

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

MAX_HISTORY_ROWS = 5000


# ==================== 按天存储的历史数据读写 ====================

def _get_daily_alpha_file():
    """获取当天Alpha历史文件路径"""
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(HISTORY_DIR, f"alpha_{date_str}.csv")

def _get_daily_basis_file():
    """获取当天基差历史文件路径"""
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(HISTORY_DIR, f"basis_{date_str}.csv")

def _get_today_alpha_files():
    """获取当天所有Alpha历史文件（可能有多个日期）"""
    return [_get_daily_alpha_file()]

def _ensure_csv_headers(filepath, headers):
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def _append_csv_row(filepath, row):
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        logger.error(f"写入历史数据失败 {filepath}: {e}")


def load_history(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            return list(csv.reader(f))
    except Exception:
        return []


# ==================== Target组合加载 ==================

def load_target_portfolio(target_file_path: str, date_str: str = None) -> Optional[pd.DataFrame]:
    """加载指定账户的 target 组合"""
    prev_date = mdata.get_previous_trading_date()
    prev_date = prev_date.replace('-', '')
    if date_str is None:
        date_str = mdata.get_today_str()

    file_path = target_file_path.format(date=date_str)
    if 'CZQZX300' in target_file_path:
        file_path = target_file_path.format(date=prev_date)

    if not os.path.exists(file_path):
        logger.warning(f"Target文件不存在: {file_path}")
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            if files:
                file_path = os.path.join(dir_path, files[0])
                logger.info(f"使用目录下找到的文件: {file_path}")
            else:
                return None
        else:
            return None

    def format_code(code):
        """将6位数字代码转换为带后缀的格式"""
        code = int(str(code)[:6])
        code = str(code).zfill(6)
        code = code + '.XSHG' if code.startswith('6') else code + '.XSHE'
        return code

    try:
        # df = pd.read_csv(file_path, dtype={"code": str})
        df = pd.read_csv(file_path)
        df = df.rename(columns={"ticker": "code", "w": "weight"})
        # if "code" in df.columns and "w" in df.columns:
        #     df = df.rename(columns={"w": "weight"})
        # elif "code" not in df.columns or "weight" not in df.columns:
        #     cols = df.columns.tolist()
        #     if len(cols) >= 2:
        #         df = df.rename(columns={cols[0]: "code", cols[1]: "weight"})
        # elif "ticker" not in df.columns or "weight" not in df.columns:
        #     cols = df.columns.tolist()
        #     if len(cols) >= 2:
        #         df = df.rename(columns={cols[0]: "code", cols[1]: "weight"})
        tw = df["weight"].sum()
        if tw > 0:
            df["weight"] = df["weight"] / tw
        # df["code"] = df["code"].str.strip()
        df["code"] = df["code"].apply(format_code)
        logger.info(f"加载Target组合成功: {len(df)}只股票")
        return df
    except Exception as e:
        logger.error(f"加载Target文件失败 {file_path}: {e}")
        return None


# ==================== 中位数涨幅计算 ==================

def calculate_median_return(codes: list) -> Optional[float]:
    """计算一批股票的中位数涨幅（基于当日涨跌幅）"""
    if not codes:
        return None
    try:
        prices = mdata.get_stock_prices(codes)
        if not prices:
            return None
        prev_date = mdata.get_previous_trading_date()
        valid_returns = []
        batch_size = 100
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            try:
                df_hist = rqdatac.get_price(batch, start_date=prev_date, end_date=prev_date,
                                            frequency="1d", fields=["close"])
                if df_hist is not None and not df_hist.empty:
                    for c in batch:
                        if c in prices and c in df_hist.index.get_level_values('order_book_id'):
                            prev_close = df_hist.loc[(c, prev_date), "close"]
                            if not np.isnan(prev_close) and prev_close > 0:
                                ret = (prices[c] - prev_close) / prev_close * 100
                                valid_returns.append(ret)
            except Exception:
                continue
        if not valid_returns:
            return None
        return round(float(np.median(valid_returns)), 4)
    except Exception as e:
        logger.warning(f"计算中位数涨幅失败: {e}")
        return None


# ==================== 账户净值加载 ==================

def load_account_net_value(account_cfg) -> Optional[float]:
    """从QMT导出文件加载账户实时净资产"""
    date_str = mdata.get_today_str()

    if account_cfg.account_type == "credit":
        file_path = os.path.join(account_cfg.qmt_dir, "Credit", f"Account-{date_str}.csv")
    else:
        file_path = os.path.join(account_cfg.qmt_dir, "Stock", f"Account-{date_str}.csv")

    if not os.path.exists(file_path):
        logger.warning(f"账户净值文件不存在: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding="gbk")

        if account_cfg.account_id:
            for col in ["资金账号", "账号", "account_id", "AccountID", "客户资金账号"]:
                if col in df.columns:
                    rows = df[df[col].astype(str).str.strip() == str(account_cfg.account_id).strip()]
                    if not rows.empty:
                        df = rows
                        break

        for col in ["净资产", "动态权益", "净值", "net_assets"]:
            if col in df.columns:
                value = float(df.iloc[0][col])
                logger.info(f"账户 {account_cfg.name}({account_cfg.account_id}) 净资产: {value:.2f}")
                return value

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            value = float(df.iloc[0][numeric_cols[0]])
            logger.info(f"账户 {account_cfg.name}({account_cfg.account_id}) 净资产(取自{numeric_cols[0]}): {value:.2f}")
            return value

        logger.warning(f"无法在文件中找到净资产列: {file_path}")
        return None
    except Exception as e:
        logger.error(f"读取账户净值文件失败 {file_path}: {e}")
        return None


def load_account_prev_net_value(account_cfg) -> Optional[float]:
    """获取账户昨日净资产"""
    try:
        prev_date = mdata.get_previous_trading_date()
        prev_date_str = prev_date.replace("-", "")

        if account_cfg.account_type == "credit":
            file_path = os.path.join(account_cfg.qmt_dir, "Credit", f"Account-{prev_date_str}.csv")
        else:
            file_path = os.path.join(account_cfg.qmt_dir, "Stock", f"Account-{prev_date_str}.csv")

        if not os.path.exists(file_path):
            logger.warning(f"昨日账户净值文件不存在: {file_path}")
            return None

        df = pd.read_csv(file_path, encoding="gbk")

        if account_cfg.account_id:
            for col in ["资金账号", "账号", "account_id", "AccountID", "客户资金账号"]:
                if col in df.columns:
                    rows = df[df[col].astype(str).str.strip() == str(account_cfg.account_id).strip()]
                    if not rows.empty:
                        df = rows
                        break

        for col in ["净资产", "动态权益", "净值"]:
            if col in df.columns:
                return float(df.iloc[0][col])
        return None
    except Exception as e:
        logger.warning(f"获取昨日净值失败: {e}")
        return None


# ==================== 核心Alpha计算 ==================

def calculate_theoretical_alpha(target_df: pd.DataFrame) -> Optional[dict]:
    """计算理论超额收益"""
    if target_df is None or target_df.empty:
        return {"portfolio_return": 0, "benchmark_return": 0, "excess_return": 0,
                "stock_count": 0, "total_weight": 0, "valid": False}

    try:
        codes = target_df["code"].tolist()
        prices = mdata.get_stock_prices(codes)

        if not prices:
            return {"portfolio_return": 0, "benchmark_return": 0, "excess_return": 0,
                    "stock_count": 0, "total_weight": 0, "valid": False}

        bench_code = CONFIG.benchmark.code
        bench_price = mdata.get_index_price(bench_code)
        bench_prev_close = mdata.get_index_prev_close(bench_code)

        if bench_price is None or bench_prev_close is None or bench_prev_close == 0:
            logger.warning("无法获取基准指数行情")
            return {"portfolio_return": 0, "benchmark_return": 0, "excess_return": 0,
                    "stock_count": 0, "total_weight": 0, "valid": False}

        benchmark_return = (bench_price - bench_prev_close) / bench_prev_close * 100
        prev_date = mdata.get_previous_trading_date()

        valid_codes = [c for c in codes if c in prices]
        if not valid_codes:
            return {"portfolio_return": 0, "benchmark_return": round(benchmark_return, 4),
                    "excess_return": round(-benchmark_return, 4),
                    "stock_count": 0, "total_weight": 0, "valid": True}

        prev_closes = {}
        batch_size = 100
        for i in range(0, len(valid_codes), batch_size):
            batch = valid_codes[i:i + batch_size]
            try:
                df_hist = rqdatac.get_price(batch, start_date=prev_date, end_date=prev_date,
                                            frequency="1d", fields=["close"])
                if df_hist is not None and not df_hist.empty:
                    codes_with_data = df_hist.index.get_level_values('order_book_id').unique()
                    for c in batch:
                        if c in codes_with_data:
                            val = df_hist.loc[(c, prev_date), "close"]
                            if not np.isnan(val):
                                prev_closes[c] = float(val)
            except Exception as e:
                logger.warning(f"获取昨日收盘价失败(batch): {e}")
                continue

        total_weight = 0
        weighted_return = 0
        stock_count = 0
        stock_returns = []  # 收集个股涨跌幅用于中位数

        for _, row in target_df.iterrows():
            code = row["code"]
            weight = row["weight"]
            if code not in prices or code not in prev_closes:
                continue
            cur_price = prices[code]
            prev_close = prev_closes[code]
            if prev_close <= 0:
                continue
            stock_return = (cur_price - prev_close) / prev_close * 100
            weighted_return += weight * stock_return
            total_weight += weight
            stock_count += 1
            stock_returns.append(stock_return)

        if total_weight <= 0:
            return {"portfolio_return": 0, "benchmark_return": round(benchmark_return, 4),
                    "excess_return": round(-benchmark_return, 4),
                    "stock_count": 0, "total_weight": 0, "valid": True}

        portfolio_return = weighted_return / total_weight
        excess_return = portfolio_return - benchmark_return
        median_return = round(float(np.median(stock_returns)), 4) if stock_returns else 0

        return {"portfolio_return": round(portfolio_return, 4),
                "benchmark_return": round(benchmark_return, 4),
                "excess_return": round(excess_return, 4),
                "stock_count": stock_count, "total_weight": round(total_weight, 4),
                "benchmark_price": bench_price, "benchmark_prev_close": bench_prev_close,
                "median_return": median_return,
                "valid": True}

    except Exception as e:
        logger.error(f"计算理论Alpha失败: {e}")
        return {"portfolio_return": 0, "benchmark_return": 0, "excess_return": 0,
                "stock_count": 0, "total_weight": 0, "valid": False}


def calculate_account_alpha(account_cfg) -> Optional[dict]:
    """计算账户实际超额收益"""
    try:
        net_value = load_account_net_value(account_cfg)
        if net_value is None:
            return {"net_value": 0, "prev_net_value": 0, "net_return": 0,
                    "benchmark_return": 0, "excess_return": 0, "valid": False}

        prev_net_value = load_account_prev_net_value(account_cfg)
        if prev_net_value is None or prev_net_value <= 0:
            return {"net_value": net_value, "prev_net_value": 0, "net_return": 0,
                    "benchmark_return": 0, "excess_return": 0,
                    "valid": True, "no_prev": True}
        deposit = account_cfg.deposit
        net_return = ((net_value + deposit) - prev_net_value) / prev_net_value * 100
        bench_code = CONFIG.benchmark.code
        bench_price = mdata.get_index_price(bench_code)
        bench_prev_close = mdata.get_index_prev_close(bench_code)

        if bench_price and bench_prev_close and bench_prev_close > 0:
            benchmark_return = (bench_price - bench_prev_close) / bench_prev_close * 100
        else:
            benchmark_return = 0

        excess_return = net_return - benchmark_return

        return {"net_value": round(net_value, 2), "prev_net_value": round(prev_net_value, 2),
                "net_return": round(net_return, 4), "benchmark_return": round(benchmark_return, 4),
                "excess_return": round(excess_return, 4), "valid": True, "no_prev": False}

    except Exception as e:
        logger.error(f"计算账户{account_cfg.name}Alpha失败: {e}")
        return {"net_value": 0, "prev_net_value": 0, "net_return": 0,
                "benchmark_return": 0, "excess_return": 0, "valid": False}


# ==================== 历史数据记录（按天） ==================

def save_alpha_history(accounts_data: list, benchmark_return: float):
    """保存Alpha历史到CSV（当天文件）"""
    alpha_file = _get_daily_alpha_file()
    headers = ["timestamp"]
    for a in accounts_data:
        headers.append(f"theoretical_{a['short_name']}_excess")
    for a in accounts_data:
        headers.append(f"account_{a['short_name']}_excess")
    headers.append("benchmark_return")
    _ensure_csv_headers(alpha_file, headers)

    now_str = datetime.now().strftime("%H:%M:%S")
    row = [now_str]
    for a in accounts_data:
        ta = a.get("theoretical", {})
        row.append(round(ta.get("excess_return", 0), 4) if ta.get("valid") else "")
    for a in accounts_data:
        ad = a.get("account", {})
        row.append(round(ad.get("excess_return", 0), 4) if ad.get("valid") else "")
    row.append(round(benchmark_return, 4))
    _append_csv_row(alpha_file, row)


def save_basis_history(basis_data: dict):
    """保存股指期货基差历史到CSV（当天文件）"""
    if not basis_data:
        return
    basis_file = _get_daily_basis_file()
    names = sorted(basis_data.keys())
    headers = ["timestamp"]
    for n in names:
        headers.append(f"{n}_basis")
    for n in names:
        headers.append(f"{n}_cost")
    _ensure_csv_headers(basis_file, headers)

    now_str = datetime.now().strftime("%H:%M:%S")
    row = [now_str]
    for n in names:
        b = basis_data[n]
        row.append(round(b.get("dominant_basis", 0), 2))
    for n in names:
        b = basis_data[n]
        row.append(round(b.get("dominant_basis_cost", 0), 4))
    _append_csv_row(basis_file, row)


def load_alpha_history_json():
    """加载当天Alpha历史，返回前端可用的JSON"""
    alpha_file = _get_daily_alpha_file()
    rows = load_history(alpha_file)
    if not rows or len(rows) < 2:
        return {"times": [], "theoretical": {}, "accounts": {}, "benchmark": []}
    headers = rows[0]
    data = rows[1:]
    n_acc = len(CONFIG.accounts)
    result = {"times": [r[0] for r in data], "theoretical": {}, "accounts": {}, "benchmark": []}

    for i in range(n_acc):
        th_key = f"theoretical_{CONFIG.accounts[i].short_name}_excess"
        ac_key = f"account_{CONFIG.accounts[i].short_name}_excess"
        result["theoretical"][th_key] = []
        result["accounts"][ac_key] = []
        for r in data:
            if 1 + i < len(r) and r[1 + i]:
                try:
                    result["theoretical"][th_key].append(float(r[1 + i]))
                except ValueError:
                    result["theoretical"][th_key].append(None)
            else:
                result["theoretical"][th_key].append(None)
            if 1 + n_acc + i < len(r) and r[1 + n_acc + i]:
                try:
                    result["accounts"][ac_key].append(float(r[1 + n_acc + i]))
                except ValueError:
                    result["accounts"][ac_key].append(None)
            else:
                result["accounts"][ac_key].append(None)

    for r in data:
        if len(r) > len(headers) - 1 and r[-1]:
            try:
                result["benchmark"].append(float(r[-1]))
            except ValueError:
                result["benchmark"].append(None)
        else:
            result["benchmark"].append(None)
    return result


def load_basis_history_json():
    """加载当天基差历史，返回前端可用的JSON"""
    basis_file = _get_daily_basis_file()
    rows = load_history(basis_file)
    if not rows or len(rows) < 2:
        return {"times": [], "series": {}}
    headers = rows[0]
    data = rows[1:]
    n_fut = (len(headers) - 1) // 2
    result = {"times": [r[0] for r in data], "series": {}}
    for i in range(n_fut):
        name = headers[1 + i].replace("_basis", "")
        result["series"][f"{name}_basis"] = []
        result["series"][f"{name}_cost"] = []
        for r in data:
            try:
                result["series"][f"{name}_basis"].append(
                    float(r[1 + i]) if 1 + i < len(r) and r[1 + i] else None)
            except (ValueError, IndexError):
                result["series"][f"{name}_basis"].append(None)
            try:
                result["series"][f"{name}_cost"].append(
                    float(r[1 + n_fut + i]) if 1 + n_fut + i < len(r) and r[1 + n_fut + i] else None)
            except (ValueError, IndexError):
                result["series"][f"{name}_cost"].append(None)
    return result


# ==================== 汇总数据 ==================

def collect_all_data() -> dict:
    """收集所有监控数据，供Web前端展示"""
    now = datetime.now()

    bench_code = CONFIG.benchmark.code
    bench_price = mdata.get_index_price(bench_code)
    bench_prev_close = mdata.get_index_prev_close(bench_code)
    bench_return = 0
    if bench_price and bench_prev_close and bench_prev_close > 0:
        bench_return = round((bench_price - bench_prev_close) / bench_prev_close * 100, 4)

    # --- 计算基准成分股中位数涨幅 ---
    benchmark_median_return = None
    try:
        bench_components = mdata.get_index_components(bench_code)
        if bench_components:
            benchmark_median_return = calculate_median_return(bench_components)
    except Exception as e:
        logger.warning(f"获取基准成分股中位数失败: {e}")

    benchmark_data = {
        "name": CONFIG.benchmark.name, "code": bench_code,
        "price": bench_price, "prev_close": bench_prev_close,
        "return_pct": bench_return,
        "median_return": benchmark_median_return
    }

    accounts_data = []
    for acct in CONFIG.accounts:
        target_df = load_target_portfolio(acct.target_file_path) if acct.target_file_path else None
        target_alpha = calculate_theoretical_alpha(target_df)
        if target_alpha and target_alpha.get("valid") and target_df is not None:
            target_alpha["stock_count"] = len(target_df)
        account_alpha = calculate_account_alpha(acct)
        acc_entry = {
            "name": acct.name,
            "short_name": acct.short_name,
            "theoretical": target_alpha if target_alpha else {"valid": False},
            "account": account_alpha if account_alpha else {"valid": False}
        }
        # 如果该账户有target文件，也计算target成分中位数涨幅
        if target_df is not None:
            target_codes = target_df["code"].tolist()
            target_median = calculate_median_return(target_codes)
            acc_entry["target_median_return"] = target_median
        accounts_data.append(acc_entry)

    basis_data = mdata.get_all_future_data()

    # 保存历史（按天）
    save_alpha_history(accounts_data, bench_return)
    save_basis_history(basis_data)

    # 加载当天历史供前端
    alpha_history = load_alpha_history_json()
    basis_history = load_basis_history_json()

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "trading_time": mdata.is_trading_time(),
        "benchmark": benchmark_data,
        "accounts": accounts_data,
        "futures_basis": basis_data,
        "history": {
            "alpha": alpha_history,
            "basis": basis_history
        }
    }
