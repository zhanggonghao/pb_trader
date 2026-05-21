
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