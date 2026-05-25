"""
数据读取模块 - 读取所有原始数据
"""
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import config


def parse_date_str(date_str):
    """将YYYYMMDD字符串转换为datetime"""
    if isinstance(date_str, int):
        date_str = str(date_str)
    return datetime.strptime(date_str, "%Y%m%d")


def date_to_str(dt):
    """将datetime转换为YYYYMMDD字符串"""
    return dt.strftime("%Y%m%d")


def date_to_plot_str(dt):
    """将datetime转换为用于绘图的YYYYMMDD字符串"""
    return dt.strftime("%Y%m%d")


class DataReader:
    def __init__(self, prev_date=None):
        self.base_dir = config.BASE_DIR
        self.report_dates = self._get_trading_dates(config.REPORT_START, config.REPORT_END)
        self.prev_date = prev_date  # 由外部通过米筐获取后传入

    def set_trading_calendar(self, trading_dates):
        """用米筐真实交易日历覆盖本地计算的日期"""
        if trading_dates:
            print(f"  使用米筐交易日历: {len(trading_dates)}个交易日")
            self.report_dates = trading_dates

    def _get_trading_dates(self, start, end):
        """获取交易日列表"""
        dates = []
        current = parse_date_str(start)
        end_dt = parse_date_str(end)
        while current <= end_dt:
            dates.append(date_to_str(current))
            current = self._next_trading_day(current)
        return dates

    def _next_trading_day(self, dt):
        """获取下一个交易日（简单跳过周末）"""
        from datetime import timedelta
        next_day = dt + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    def get_trading_dates_with_prev(self):
        """获取包含前一天的交易日期列表"""
        return [self.prev_date] + self.report_dates

    def load_net_email_nav(self, date_str):
        """从净值邮件Excel读取产品净值（自动匹配文件名）"""
        # 先用精确文件名匹配
        exact_filename = f"【基金净值】SAXM36_配邦恒升中性1号私募证券投资基金_{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.xlsx"
        filepath = os.path.join(config.NET_EMAIL_DIR, exact_filename)
        if os.path.exists(filepath):
            return self._read_nav_from_excel(filepath)

        # 失败后用 glob 模糊匹配：按日期查找目录下所有 xlsx
        patterns = [
            f"*{date_str}*.xlsx",
            f"*{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}*.xlsx",
            f"*{date_str[:4]}{date_str[4:6]}{date_str[6:]}*.xlsx",
        ]
        for pattern in patterns:
            search_path = os.path.join(config.NET_EMAIL_DIR, pattern)
            files = sorted(glob.glob(search_path))
            if files:
                return self._read_nav_from_excel(files[0])

        return None

    def _read_nav_from_excel(self, filepath):
        """从Excel中读取净值数值"""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(filepath)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            if len(rows) > 1 and rows[1][0] is not None:
                return float(rows[1][3])
        except:
            pass
        return None

    def load_stk_account_info(self, date_str, account_key):
        """读取证券账户信息"""
        pattern = os.path.join(config.STK_FUT_DIR, date_str, f"{date_str}_{account_key}_*_stk_info.csv")
        files = glob.glob(pattern)
        if not files:
            return None
        try:
            df = pd.read_csv(files[0])
            return df.iloc[0]
        except:
            return None

    def load_fut_info(self, date_str):
        """读取期货账户信息"""
        filepath = os.path.join(config.STK_FUT_DIR, date_str, f"{date_str}_PBHSZX1H_fut_info.csv")
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            return df.iloc[0]
        except:
            return None

    def load_positions(self, date_str, account_key):
        """读取持仓数据"""
        pattern = os.path.join(config.POS_DIR, date_str, f"{date_str}_{account_key}_*_pos.csv")
        files = glob.glob(pattern)
        if not files:
            return None
        try:
            df = pd.read_csv(files[0])
            return df
        except:
            return None

    def find_account_keys(self, product_code, date_str):
        """根据产品代码找到所有相关账户"""
        search_pattern = os.path.join(config.STK_FUT_DIR, date_str, f"{date_str}_*_{product_code}_stk_info.csv")
        files = glob.glob(search_pattern)
        accounts = []
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if len(parts) >= 3:
                account_key = parts[1]
                if account_key not in accounts:
                    accounts.append(account_key)
        return accounts

    def load_all_positions_for_date(self, date_str):
        """加载某日所有账户的持仓合并"""
        accounts = self.find_account_keys(config.REPORT_PRODUCT_CODE, date_str)
        all_positions = []
        for acc in accounts:
            pos_df = self.load_positions(date_str, acc)
            if pos_df is not None and not pos_df.empty:
                pos_df['account'] = acc
                all_positions.append(pos_df)
        if all_positions:
            return pd.concat(all_positions, ignore_index=True)
        return None

    def collect_daily_data(self):
        """收集所有日期的数据"""
        data = {
            'dates': self.report_dates,
            'prev_date': self.prev_date,
            'nav': {},
            'stk_accounts': {},
            'fut': {},
        }

        accounts = self.find_account_keys(config.REPORT_PRODUCT_CODE, self.report_dates[0])
        data['accounts'] = accounts

        for d in self.report_dates:
            data['nav'][d] = self.load_net_email_nav(d)
            data['stk_accounts'][d] = {}
            for acc in accounts:
                data['stk_accounts'][d][acc] = self.load_stk_account_info(d, acc)
            data['fut'][d] = self.load_fut_info(d)

        data['nav'][self.prev_date] = self.load_net_email_nav(self.prev_date)
        data['stk_accounts'][self.prev_date] = {}
        for acc in accounts:
            data['stk_accounts'][self.prev_date][acc] = self.load_stk_account_info(self.prev_date, acc)
        data['fut'][self.prev_date] = self.load_fut_info(self.prev_date)

        return data

    def collect_positions_data(self):
        """收集所有日期的持仓数据"""
        positions = {}
        for d in self.report_dates:
            pos_df = self.load_all_positions_for_date(d)
            if pos_df is not None:
                positions[d] = pos_df
        return positions

    def load_factor_exposure_from_parquet(self, date_str, factor_type='stock'):
        """
        从parquet文件读取因子暴露数据
        
        参数:
            date_str: 日期字符串 (YYYYMMDD)
            factor_type: 'stock' 或 'benchmark'
        
        返回:
            DataFrame: 因子暴露数据，列包含 [code, lcap_exposure, liquidity_exposure, beta_exposure, ...]
        """
        try:
            if factor_type == 'benchmark':
                filepath = config.FactorConfig.BENCHMARK_FACTOR_PATH
            else:
                filepath = config.FactorConfig.STOCK_FACTOR_PATH
            
            if not os.path.exists(filepath):
                print(f"  因子文件不存在: {filepath}")
                return None
            
            # 读取parquet文件
            df = pd.read_parquet(filepath)
            
            # 将日期列转换为字符串格式进行筛选
            if 'date' in df.columns:
                # 转换日期列为字符串 YYYYMMDD
                df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
                df_filtered = df[df['date_str'] == date_str].copy()
            else:
                print(f"  因子文件缺少date列")
                return None
            
            if df_filtered.empty:
                print(f"  未找到 {date_str} 的因子数据")
                return None
            
            # 对于股票因子，将 order_book_id 转换为 code 格式（去掉交易所后缀）
            if factor_type == 'stock' and 'order_book_id' in df_filtered.columns:
                df_filtered['code'] = df_filtered['order_book_id'].str.replace(r'\.XSH[EG]', '', regex=True)
            
            # 只保留需要的因子列
            selected_factors = list(config.FactorConfig.SELECTED_FACTORS.keys())
            # 检查因子列是否存在（code列是必需的，但不是因子列）
            available_factor_cols = [c for c in selected_factors if c in df_filtered.columns]
            missing_factors = set(selected_factors) - set(available_factor_cols)
            
            if missing_factors:
                print(f"  缺少因子列: {missing_factors}")
            
            # 返回包含code（如果存在）和所有可用因子列的数据
            return_cols = available_factor_cols.copy()
            if 'code' in df_filtered.columns:
                return_cols = ['code'] + return_cols
            return df_filtered[return_cols]
            
        except Exception as e:
            print(f"  读取因子文件失败: {e}")
            return None

    def calculate_portfolio_factor_exposure(self, date_str, positions_df):
        """
        计算组合持仓的加权因子暴露
        
        参数:
            date_str: 日期字符串
            positions_df: 持仓DataFrame，包含 [code, hold, market_value]
        
        返回:
            dict: 各因子的加权暴露值
        """
        # 读取个股因子暴露
        factor_df = self.load_factor_exposure_from_parquet(date_str, factor_type='stock')
        if factor_df is None or positions_df is None:
            return None
        
        # 将持仓数据的 code 列转换为字符串类型，确保与因子数据类型一致
        positions_df_copy = positions_df.copy()
        positions_df_copy['code'] = positions_df_copy['code'].astype(str)
        factor_df['code'] = factor_df['code'].astype(str)
        
        # 合并持仓和因子数据
        merged = positions_df_copy.merge(factor_df, on='code', how='inner')
        if merged.empty:
            print(f"  {date_str}: 持仓与因子数据无匹配")
            return None
        
        # 计算市值权重
        if 'market_value' not in merged.columns:
            merged['market_value'] = merged['hold']  # 如果没有市值，用持仓量代替
        
        total_mv = merged['market_value'].sum()
        if total_mv <= 0:
            return None
        
        merged['weight'] = merged['market_value'] / total_mv
        
        # 计算加权因子暴露
        result = {}
        selected_factors = list(config.FactorConfig.SELECTED_FACTORS.keys())
        for factor in selected_factors:
            if factor in merged.columns:
                result[factor] = (merged[factor] * merged['weight']).sum()
        
        return result

    def get_benchmark_factor_exposure(self, date_str):
        """
        获取基准指数的因子暴露
        
        参数:
            date_str: 日期字符串
        
        返回:
            dict: 基准各因子的暴露值
        """
        factor_df = self.load_factor_exposure_from_parquet(date_str, factor_type='benchmark')
        if factor_df is None or factor_df.empty:
            return None
        
        # 基准数据通常只有一行或需要取平均
        result = {}
        selected_factors = list(config.FactorConfig.SELECTED_FACTORS.keys())
        for factor in selected_factors:
            if factor in factor_df.columns:
                result[factor] = factor_df[factor].mean()
        
        return result
