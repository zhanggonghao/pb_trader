import os
import pickle
import time
import pandas as pd
import numpy as np
import datetime as dt
import rqdatac

rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=")

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


class DealData(object):
    def __init__(self, date):
        self.date = date
        self.pre_date = rqdatac.get_previous_trading_date(
            date).strftime('%Y%m%d')
        self.target = None
        self.data_dic = {}

    # 获取各期货合约的年化成本

    def get_min_fut_cost(self):
        fut_mkt_df = rqdatac.all_instruments(
            type='Future', market='cn', date=self.date)
        fut_mkt_df = fut_mkt_df[(fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM'))) & (
            fut_mkt_df['symbol'].apply(lambda x: '连续' not in x))].reset_index(drop=True)
        # print(fut_mkt_df)
        fut_mkt_codes = fut_mkt_df['order_book_id'].tolist()

        try:
            cur_data = rqdatac.current_minute(
                fut_mkt_codes, fields=None, skip_suspended=False).reset_index()
            cur_data = pd.merge(cur_data, fut_mkt_df[[
                'order_book_id', 'contract_multiplier']], on='order_book_id')
            # print(cur_data)

            return cur_data
        except:
            return -1

    def main(self):
        ddp = DateDealProcess()
        if ddp.judge_trading_date(self.date):
            # self.get_min_fut_cost()
            file_path = rf'E:\code\generate_split_system\data\raw\real_minute\{self.date}_data.csv'

            time_hhmm = []

            while True:
                hhmm = dt.datetime.now().strftime('%H%M')

                if hhmm < '0931':
                    print('未到开盘时间')
                    time.sleep(1)
                    continue
                elif hhmm > '1501':
                    print('已收盘，结束程序')
                    break

                # --- 处理分钟数据 ---
                if hhmm not in time_hhmm:
                    time_hhmm.append(hhmm)
                    print(f'计算 {hhmm} 数据: {dt.datetime.now().strftime("%H%M%S")}')
                    # tmp = self.get_min_fut_cost()
                    # 引入尝试机制
                    max_retries = 3
                    delay = 1
                    for attempt in range(1, max_retries + 1):
                        try:
                            tmp = self.get_min_fut_cost()
                        except Exception as e:
                            print(f"第 {attempt} 次尝试失败: {e}")
                            if attempt == max_retries:
                                print("已达到最大重试次数，放弃。")
                                raise  # 重新抛出最后一次异常
                            else:
                                print(f"等待 {delay} 秒后重试...")
                                time.sleep(delay)

                    # 保存到文件
                    if not os.path.exists(file_path):
                        # os.makedirs(file_path)
                        tmp.to_csv(file_path)
                        # time.sleep(1)
                    else:
                        data = pd.read_csv(file_path, index_col=0)
                        data = pd.concat([data, tmp]).reset_index(drop=True)
                        data.to_csv(file_path)

                        # time.sleep(5)
                else:
                    # print(f'当前时间 {hhmm} 已存入或不在交易时段')
                    time.sleep(60)


if __name__ == "__main__":
    date = dt.datetime.now().strftime('%Y%m%d')
    DD = DealData(date)
    DD.main()
