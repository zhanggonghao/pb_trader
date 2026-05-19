import os
import yaml
import datetime as dt
from ultis.email_manager import *
import rqdatac
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)


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
ddp = DateDealProcess()
if ddp.judge_trading_date(date):

    root_path = os.path.dirname(os.path.abspath(__file__))
    with open(f'{root_path}/target_config.yaml', 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)

    model_path = config.get('model_path')
    pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
    model_file = f'df_test_PB_ScorpioV4_2020_{pre_date}.parquet'
    model_paths = f'{model_path}/{model_file}'

    if not os.path.exists(model_paths): 
        manager = EmailManager()
        manager.send_email_with_attachments(['15556235305@163.com', 'pagududeshengjiang@shpbjj.com', 'xu_hengsheng@163.com', 'annie040216@163.com'], f'{date}-缺少评分表', f'{model_path}目录下未发现{model_file}文件')
        manager.logout()

