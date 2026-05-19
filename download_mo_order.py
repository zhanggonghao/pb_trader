import os
import pandas as pd
import datetime as dt
from ultis.FileTransfer import *
from pathlib import Path
import shutil


# date = ''
date = dt.datetime.now().strftime('%Y%m%d')

remote_path = f'/home/zhanggh/DailyScripts/SplitSystemData/mo_order/{date}'

loc_path = r'E:\mo_order'
loc_files = os.listdir(loc_path)
if len(loc_files) != 0:
    for loc_file in loc_files:
        os.remove(f'{loc_path}/{loc_file}')

transfer = LinuxFileTransfer()
# 链接服务器
transfer.connect()

transfer.download_remote_files(remote_path, loc_path)


# 断开连接
transfer.disconnect()

