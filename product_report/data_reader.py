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
    def __init__(self):
        self.base_dir = config.BASE_DIR
        self.report_dates = self._get_trading_dates(config.REPORT_START, config.REPORT_END)
        self.prev_date = config.REPORT_PREV

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
        """从净值邮件Excel读取产品净值"""
        filename = f"【基金净值】SAXM36_配邦恒升中性1号私募证券投资基金_{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.xlsx"
        filepath = os.path.join(config.NET_EMAIL_DIR, filename)
        if not os.path.exists(filepath):
            return None
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
