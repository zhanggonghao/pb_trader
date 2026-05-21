import sys
import os
import yaml
from pathlib import Path
# 自动把 D:\code 加入 Python 搜索路径
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import datetime as dt
from ultis.email_manager import *
import rqdatac

rqdatac.init(username="license",
             password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=",
             use_pool=True, max_pool_size=8)


def format_percent(x: float) -> str:
    return f"{x:.2%}" if pd.notnull(x) else "--"


def format_money(x: float, unit: str = '万') -> str:
    if unit == '万':
        return f"{x / 10000:,.2f}"
    return f"{x:,.2f}"


class DateDealProcess:
    def __init__(self):
        pass

    def get_trading_lst(self, start_date='20000101', end_date=''):
        dates = rqdatac.get_trading_dates(start_date, end_date, market='cn')
        dates = [i.strftime('%Y%m%d') for i in dates]
        return dates

    # 判断是否为交易日
    def judge_trading_date(self, date) -> bool:
        dates = self.get_trading_lst(end_date=date)
        return date in dates


date = dt.datetime.now().strftime('%Y%m%d')
# date = '20260421'
ddp = DateDealProcess()
if ddp.judge_trading_date(date):
    path = rf'E:\code\generate_split_system\data\out\stk_fut\{date}\{date}_gtht_PBHSZX1H_stk_info.csv'
    fut_path = rf'E:\code\generate_split_system\data\out\stk_fut\{date}\{date}_PBHSZX1H_fut_info.csv'
    net_path = rf'E:\code\generate_split_system\data\out\net_data\{date}\{date}_PBHSZX1H_net_info.csv'
    data = pd.read_csv(path)
    fut_data = pd.read_csv(fut_path)
    net_data = pd.read_csv(net_path)
    # print(fut_data)
    # print(net_data)
    data['期货收益1'] = fut_data.loc[0, 'quant收益'] * (data.loc[0, '多头持仓市值'] / net_data.loc[0, '证券户持仓市值'])
    print(data)
    model_paths = rf'E:\code\generate_split_system\data\conv\czq_data\{date}_zx300_gtht_PBHSZX1H_info.csv'
    data.to_csv(model_paths, encoding='gbk', index=False)

    # 持仓文件
    pos_paths = rf'E:\code\generate_split_system\data\standarddata\pos\{date}\{date}_gtht_PBHSZX1H_pos.csv'

    display_df = data[
        ['日期', '产品名', '账户名', '证券户净资产', '多头持仓市值', '证券户总收益', '基准收益率', '多头超额',
         '多头超额(含T0)', '期货收益1']].copy()
    display_df['证券户净资产'] = display_df['证券户净资产'].apply(lambda x: format_money(x))
    display_df['多头持仓市值'] = display_df['多头持仓市值'].apply(lambda x: format_money(x))
    display_df['证券户总收益'] = display_df['证券户总收益'].apply(lambda x: format_money(x))
    display_df['期货收益1'] = display_df['期货收益1'].apply(lambda x: format_money(x))
    display_df['基准收益率'] = display_df['基准收益率'].apply(format_percent)
    display_df['多头超额'] = display_df['多头超额'].apply(format_percent)
    display_df['多头超额(含T0)'] = display_df['多头超额(含T0)'].apply(format_percent)
    html_table = display_df.to_html(index=False, classes='dataframe', border=1)

    summary = f"""
    {date} 数据摘要

    """

    # 3. 构建完整的HTML邮件正文
    email_content = f"""
    <html>
    <head>
    <style>
        .dataframe {{ border-collapse: collapse; width: 100%; }}
        .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .dataframe th {{ background-color: #f2f2f2; }}
    </style>
    </head>
    <body>
    <h2>Excel 内容摘要</h2>
    <pre>{summary}</pre>
    <h2>数据预览(金额单位：万元)</h2>
    {html_table}
    <p>完整内容请参考附件</p>
    </body>
    </html>
    """

    if os.path.exists(model_paths):
        manager = EmailManager()
        # manager.send_email_with_attachments(['pagududeshengjiang@shpbjj.com'], f'{date}_账户收益信息', email_content, attachments=[model_paths, pos_paths], is_html=True)
        manager.send_email_with_attachments(
            ['chenzq_nju@126.com', 'pagududeshengjiang@shpbjj.com', 'wyd@pb-invests.com'], f'{date}_账户收益信息',
            email_content, attachments=[model_paths, pos_paths], is_html=True)
        manager.logout()


