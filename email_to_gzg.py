import os
import pandas as pd
import datetime as dt
from ultis.email_manager import *
import rqdatac
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)


def get_trading_lst(start_date='20260508', end_date=''):
    dates = rqdatac.get_trading_dates(start_date, end_date, market='cn')
    dates = [i.strftime('%Y%m%d') for i in dates]
    return dates

# 判断是否为交易日
def judge_trading_date(date) -> bool:
    dates = get_trading_lst(end_date=date)
    return date in dates

date = dt.datetime.now().strftime('%Y%m%d')
if judge_trading_date(date):
    pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
    try:
        remote_path = rf'\\192.168.1.168\samba\Market\trading_pred_df\df_test_PB_V0422_{pre_date}.parquet'
        data = pd.read_parquet(remote_path)
    except:
        remote_path = rf'\\192.168.1.168\samba\trading_pred_df\df_test_PB_V0422_{pre_date}.parquet'
        data = pd.read_parquet(remote_path)
    data = data.sort_values(by='avg_rank_stable', ascending=False)
    target_data = data[:10].copy()
    target_codes = target_data['order_book_id'].tolist()
    manager = EmailManager()
    manager.send_email_with_attachments(['pagududeshengjiang@shpbjj.com', '1095711734@qq.com'], f'{date} 评分排名前十股票列表', f'{target_codes}')



