import logging
import os
import sys
import yaml
import pandas as pd
import numpy as np
from ultis.FileTransfer import *
import datetime as dt
from glob import glob
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 初始化 rqdatac
import rqdatac
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=")


# 项目根目录
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
logger.info(f"项目根目录: {ROOT_PATH}")

# 匹配文件辅助函数
def match_file(path, key, date):
    """匹配指定日期和关键字的文件（最新版本）"""
    pattern = os.path.join(path, f"{key}{date}*.xlsx")
    files = sorted(glob(pattern))
    return files[-1] if files else None

def match_file_v2(path, key, date):
    """匹配包含日期和关键字的文件"""
    pattern = os.path.join(path, f"*{date}*")
    for file in glob(pattern):
        if key in file:
            return file
    return None

def match_folders(path, key):
    """匹配包含关键字的文件夹"""
    return [f for f in os.listdir(path) if key in f and os.path.isdir(os.path.join(path, f))]


class StandardTradeData:
    def __init__(self, date):
        self.date = date
        self._date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"  # 用于文件名中的日期格式
        self.config = self._load_config()
        self.standard_path = self.config.get("standard_path")
        self.product_info = self.config.get("product_info", {})
        self.transfer = LinuxFileTransfer()

    def _load_config(self):
        """加载配置文件（从远程下载或本地读取）"""
        config_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/split_config")
        os.makedirs(config_dir, exist_ok=True)
        local_config = os.path.join(config_dir, f"{self.date}_split_system.yaml")

        if not os.path.exists(local_config):
            self._download_config(local_config)

        with open(local_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _download_config(self, local_path):
        """从远程服务器下载配置文件"""
        logger.info(f"下载配置文件到 {local_path}")
        transfer = LinuxFileTransfer()
        try:
            transfer.connect()
            transfer.download_file(
                "/home/zhanggh/DailyScripts/split_system.yaml",
                local_path
            )
        finally:
            transfer.disconnect()

    def _save_dataframe(self, df, local_path):
        """保存 DataFrame 到本地（自动创建目录）"""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)

    def _read_csv_safe(self, path, **kwargs):
        """安全读取 CSV，捕获异常"""
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            logger.warning(f"读取文件失败 {path}: {e}")
            return None

    def _read_excel_safe(self, path, **kwargs):
        """安全读取 Excel，捕获异常"""
        try:
            return pd.read_excel(path, **kwargs)
        except Exception as e:
            logger.warning(f"读取 Excel 失败 {path}: {e}")
            return None

    def get_qmt_data(self):
        """获取 QMT 导出的原始数据并保存"""
        qmt_path = "E:/qmt_auto_export"
        entrust_dfs, pos_dfs, assets_dfs = [], [], []

        for acct_dir in os.listdir(qmt_path):
            acct_full = os.path.join(qmt_path, acct_dir)
            if not os.path.isdir(acct_full):
                continue
            stock_dir = os.path.join(acct_full, "Stock")
            credit_dir = os.path.join(acct_full, "Credit")

            # 委托记录
            order_file = os.path.join(stock_dir, f"Order-{self.date}.csv")
            cols = ['委托日期', '资金账号', '报单来源', '下单方式', '证券代码', '委托时间',
                    '委托价格', '委托量', '委托状态', '成交数量', '成交均价', '买卖标记',
                    '委托类别', '合同编号']
            if os.path.exists(order_file):
                df = self._read_csv_safe(order_file, encoding="gbk")
                if df is not None:
                    entrust_dfs.append(df[cols])

            credit_order_file = os.path.join(credit_dir, f"Order-{self.date}.csv")
            if os.path.exists(credit_order_file):
                df = self._read_csv_safe(credit_order_file, encoding="gbk")
                if df is not None:
                    entrust_dfs.append(df[cols])

            # 持仓
            pos_file = os.path.join(stock_dir, f"PositionStatics-{self.date}.csv")
            if os.path.exists(pos_file):
                df = self._read_csv_safe(pos_file, encoding="gbk")
                if df is not None:
                    pos_dfs.append(df[['资金账号', '证券代码', '证券名称', '当前拥股', '可用数量']])

            credit_pos_file = os.path.join(credit_dir, f"PositionStatics-{self.date}.csv")
            if os.path.exists(credit_pos_file):
                df = self._read_csv_safe(credit_pos_file, encoding="gbk")
                if df is not None:
                    pos_dfs.append(df[['资金账号', '证券代码', '证券名称', '当前拥股', '可用数量']])

            # 资金信息
            account_file = os.path.join(stock_dir, f"Account-{self.date}.csv")
            if os.path.exists(account_file):
                df = self._read_csv_safe(account_file, encoding="gbk")
                if df is not None:
                    assets_dfs.append(df)

            credit_account_file = os.path.join(credit_dir, f"Account-{self.date}.csv")
            if os.path.exists(credit_account_file):
                df = self._read_csv_safe(credit_account_file, encoding="gbk")
                if df is not None:
                    assets_dfs.append(df)

        entrust = pd.concat(entrust_dfs, ignore_index=True) if entrust_dfs else pd.DataFrame()
        pos = pd.concat(pos_dfs, ignore_index=True) if pos_dfs else pd.DataFrame()
        assets = pd.concat(assets_dfs, ignore_index=True) if assets_dfs else pd.DataFrame()

        raw_qmt_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_qmt_data", self.date)
        self._save_dataframe(entrust, os.path.join(raw_qmt_dir, f"{self.date}_raw_qmt_entrust.csv"))
        self._save_dataframe(pos, os.path.join(raw_qmt_dir, f"{self.date}_raw_qmt_pos.csv"))
        self._save_dataframe(assets, os.path.join(raw_qmt_dir, f"{self.date}_raw_qmt_assets.csv"))

        return entrust, pos, assets

    def get_zhaoshangDMA_data(self):
        """获取招商 DMA 数据"""
        dma_path = "E:\招商DMA持仓"
        order_file = match_file(dma_path, "多空收益互换订单", self._date)
        if not order_file:
            logger.warning("未找到招商 DMA 订单文件")
            return

        order = self._read_excel_safe(order_file)
        if order is None:
            return

        # 处理多头订单
        lng_order = order[~order['交易对代码'].str.startswith('I')].reset_index(drop=True)
        lng_order['code'] = lng_order['交易对代码'].apply(lambda x: int(str(x)[:6]))
        lng_order['委托时间'] = lng_order['委托时间'].apply(lambda x: str(x).split(' ')[1].split('.')[0])
        lng_order['order_price'] = lng_order['价格']
        lng_order['order_vol'] = lng_order['委托数量']
        lng_order['algo'] = 'adj'
        lng_order['vendor'] = 'HX'
        adj_start = self.product_info.get('PBTZ2H', {}).get('acct_info', {}).get('adj_algo_starttime')
        adj_end = self.product_info.get('PBTZ2H', {}).get('acct_info', {}).get('adj_algo_endtime')
        lng_order['time_period'] = f"{adj_start}-{adj_end}"
        lng_order = lng_order.rename(columns={
            '委托日期': 'date',
            '委托方向': 'dir',
            '委托时间': 'order_time',
            '委托状态': 'order_status',
            '成交数量': 'filled_vol',
            '成交均价': 'filled_price'
        })
        lng_order['dir'] = lng_order['dir'].apply(lambda x: 1 if x == '买入' else -1)
        lng_order['order_type'] = '收益互换'
        lng_order = lng_order[['date', 'code', 'dir', 'order_type', 'order_time', 'order_price',
                               'order_vol', 'order_status', 'filled_vol', 'filled_price', 'algo', 'vendor', 'time_period']]

        # 处理空头订单（期货）
        sht_order = order[order['交易对代码'].str.startswith('I')].reset_index(drop=True)
        sht_order['交易对代码'] = sht_order['交易对代码'].apply(lambda x: str(x).split('.')[0])
        sht_order['order_type'] = 'hand'
        sht_order['委托时间'] = sht_order['委托时间'].apply(lambda x: str(x).split(' ')[1].split('.')[0])
        sht_order['Direction'] = sht_order['委托方向'].apply(lambda x: -1 if x == '卖出' else 1)
        sht_order['UsedFee'] = 0
        sht_order = sht_order.rename(columns={
            '委托日期': 'date',
            '交易对代码': 'code',
            '委托时间': 'order_time',
            '价格': 'order_price',
            '委托数量': 'order_vol',
            '委托状态': 'OrderStatus',
            '成交数量': 'filled_vol',
            '成交均价': 'filled_price'
        })
        sht_order['user'] = 'quant'
        sht_order = sht_order[['date', 'code', 'order_type', 'order_time', 'Direction',
                               'order_price', 'order_vol', 'OrderStatus', 'filled_vol', 'filled_price', 'UsedFee', 'user']]

        # 持仓
        pos_file = match_file(dma_path, "多空收益互换存续合约", self._date)

        if not pos_file:
            logger.warning("未找到招商DMA持仓文件")
            return
        pos = self._read_excel_safe(pos_file)
        if pos is None:
            return
        pos_lng = pos[pos['持仓方向'] == '多头'].reset_index(drop=True)
        pos_lng = pos_lng[['资金账号', '交易对代码', '标的名称', '持仓方向', '持仓数量', '可平数量']]

        pos_sht = pos[pos['持仓方向'] == '空头'].reset_index(drop=True)
        pos_sht['交易对代码'] = pos_sht['交易对代码'].apply(lambda x: str(x).split('.')[0])
        pos_sht['Direction'] = pos_sht['持仓方向'].apply(lambda x: -1 if x == '空头' else 1)
        pos_sht['UsedMargin'] = 0
        pos_sht['amount'] = pos_sht['Direction'] * pos_sht['最新价'] * pos_sht['持仓数量'] * 300
        pos_sht = pos_sht.rename(columns={'交易对代码': 'code', '持仓数量': 'vol'})
        pos_sht['user'] = 'quant'
        pos_sht = pos_sht[['code', 'Direction', 'user', 'UsedMargin', 'vol', 'amount']]

        # 资金信息
        assets_file = match_file(dma_path, "多空收益互换资金", self._date)
        if not assets_file:
            logger.warning("未找到招商DMA资金文件")
            return
        assets = self._read_excel_safe(assets_file)
        if assets is None:
            return
        risk_file = os.path.join(dma_path, f"风险监控-风险监控_{self.date}.xlsx")
        if not os.path.exists(risk_file):
            logger.warning("未找到招商DMA风险监控文件")
            return
        risk_data = self._read_excel_safe(risk_file)
        if risk_data is None:
            return

        assets = pd.merge(assets, risk_data, left_on='资金账号', right_on='组合')
        assets['date'] = assets['更新日期']
        assets['acct_id'] = assets['资金账号']
        assets['net_assets'] = assets['账户权益']
        assets['tot_assets'] = 0
        assets['tot_debat'] = 0
        assets['ava_cash'] = assets['可用']
        assets['stk_amount'] = assets['多头名义本金']
        assets['bond_amount'] = 0
        assets['fund_amount'] = 0
        assets['reop_amount'] = 0
        assets['coll_cash'] = assets['可用保证金']
        assets['rongzi_amount'] = 0
        assets['rongquan_amount'] = 0
        assets['rqmc_amount'] = 0
        assets = assets[['date', 'acct_id', 'net_assets', 'tot_assets', 'tot_debat', 'ava_cash', 'stk_amount',
                         'bond_amount', 'fund_amount', 'reop_amount', 'coll_cash', 'rongzi_amount', 'rongquan_amount', 'rqmc_amount']]

        # 保存
        raw_other_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_other_data", self.date)
        os.makedirs(raw_other_dir, exist_ok=True)
        self._save_dataframe(lng_order, os.path.join(raw_other_dir, f"{self.date}_raw_zhaoshang_PBTZ2H_entrust.csv"))
        if pos_lng is not None:
            self._save_dataframe(pos_lng, os.path.join(raw_other_dir, f"{self.date}_raw_zhaoshang_PBTZ2H_pos.csv"))
        if assets is not None:
            self._save_dataframe(assets, os.path.join(raw_other_dir, f"{self.date}_raw_zhaoshang_PBTZ2H_assets.csv"))

        fut_data_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/fut_data", self.date)
        os.makedirs(fut_data_dir, exist_ok=True)
        self._save_dataframe(sht_order, os.path.join(fut_data_dir, f"{self.date}_PBTZ2H_fut_order.csv"))
        if pos_sht is not None:
            self._save_dataframe(pos_sht, os.path.join(fut_data_dir, f"{self.date}_PBTZ2H_fut_pos.csv"))

    def _process_futures_common(self, fut_order, fut_pos, fut_assets, product):
        """处理期货数据的公共逻辑"""
        fut_data_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/fut_data", self.date)
        fut_info = self.product_info.get(product, {}).get('fut_info')
        if not fut_info:
            return

        fut_accts = list(fut_info.values())
        df_order = fut_order[fut_order['InvestorID'].astype(str).isin(fut_accts)].copy()
        if not df_order.empty:
            df_order['order_type'] = df_order['UserCustom2'].apply(lambda x: 'shift' if pd.notnull(x) else 'hand')
            df_order['code'] = df_order['InstrumentID']
            df_order = df_order[['TradingDay', 'code', 'order_type', 'InsertTime', 'Direction', 'LimitPrice', 'Volume', 'OrderStatus', 'VolumeTraded', 'TradeAmnt', 'UsedFee', 'UserProductInfo']].rename(columns={
                'TradingDay': 'date',
                'InsertTime': 'order_time',
                'LimitPrice': 'order_price',
                'Volume': 'order_vol',
                'VolumeTraded': 'filled_vol',
                'TradeAmnt': 'filled_price',
                'UserProductInfo': 'user',
            })
            df_order['filled_vol'] = df_order['filled_vol'].astype(float)
            df_order['filled_price'] = np.where(df_order['filled_vol'] == 0, 0,
                                                df_order['filled_price'] / df_order['filled_vol'])
            df_order['user'] = df_order['user'].apply(lambda x: 'quant' if str(x).startswith('Infini') else 'cta')
            df_order['Direction'] = df_order['Direction'].apply(lambda x: -1 if x == 1 else 1)
            # print(df_order)

            # 合并招商 DMA 的空头订单（仅 PBTZ2H）
            if product == 'PBTZ2H':
                sht_order_file = os.path.join(fut_data_dir, f"{self.date}_PBTZ2H_fut_order.csv")
                if os.path.exists(sht_order_file):
                    sht_order = pd.read_csv(sht_order_file)
                    df_order = pd.concat([df_order, sht_order], ignore_index=True)
            self._save_dataframe(df_order, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_order.csv"))

        df_pos = fut_pos[fut_pos['InvestorID'].astype(str).isin(fut_accts)].copy()
        if not df_pos.empty:
            df_pos['code'] = df_pos['InstrumentID']
            df_pos['VolumeMultiple'] = df_pos['InstrumentID'].apply(lambda x: 300 if x.startswith(('IF', 'IH')) else 200)
            df_pos['PositionCost'] = df_pos['Volume'] * df_pos['SettlementPrice'] * df_pos['VolumeMultiple']
            df_pos['user'] = df_pos['MarginRateByMoney'].apply(lambda x: 'quant' if x == 0.12 else 'cta')
            df_pos = df_pos[['code', 'Direction', 'Margin', 'Volume', 'PositionCost', 'user']].rename(columns={
                'Position': 'vol',
                'PositionCost': 'amount',
                'Margin': 'UsedMargin',
                'Volume': 'vol',
            })
            df_pos = df_pos.groupby(['code', 'Direction', 'user']).sum().reset_index()
            df_pos['Direction'] = df_pos['Direction'].apply(lambda x: -1 if x == 1 else 1)
            df_pos['amount'] = df_pos['amount'] * df_pos['Direction']

            if product == 'PBTZ2H':
                sht_pos_file = os.path.join(fut_data_dir, f"{self.date}_PBTZ2H_fut_pos.csv")
                if os.path.exists(sht_pos_file):
                    sht_pos = pd.read_csv(sht_pos_file)
                    df_pos = pd.concat([df_pos, sht_pos], ignore_index=True)
            self._save_dataframe(df_pos, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_pos.csv"))

        df_assets = fut_assets[fut_assets['InvestorID'].astype(str).isin(fut_accts)].copy()
        if not df_assets.empty:
            df_assets = df_assets[['TradingDay', 'DynamicRights', 'Margin', 'Available', 'Deposit', 'Withdraw']].rename(columns={
                'TradingDay': 'date',
            })
            self._save_dataframe(df_assets, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_assets.csv"))

    def _process_hand_futures(self, product):
        """处理手动期货数据（PBZS1H）"""
        hand_dir = "E:/future_hand"
        base_name = f"{self.date}_{product}_rdqh"
        files = {
            'order': os.path.join(hand_dir, f"{base_name}_order_report.csv"),
            'pos': os.path.join(hand_dir, f"{base_name}_pos_report.csv"),
            'assets': os.path.join(hand_dir, f"{base_name}_assets_report.csv")
        }
        fut_data_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/fut_data", self.date)
        if os.path.exists(files['order']) and os.path.getsize(files['order']) > 0:
            order = self._read_csv_safe(files['order'], encoding='gbk')
            if order is not None:
                order['date'] = self.date
                order['order_type'] = 'hand'
                order['filled_vol'] = order['手数'] - order['未成交']
                order['UsedFee'] = 0
                order['user'] = 'quant'
                order = order[['date', '委托合约', 'order_type', '报单时间', '买卖', '报单价', '手数', '挂单状态', 'filled_vol', '成交均价', 'UsedFee', 'user']].rename(columns={
                    '委托合约': 'code',
                    '报单时间': 'order_time',
                    '买卖': 'Direction',
                    '报单价': 'order_price',
                    '手数': 'order_vol',
                    '成交均价': 'filled_price',
                })
                order['Direction'] = order['Direction'].apply(lambda x: -1 if '卖' in x else 1)
                self._save_dataframe(order, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_order.csv"))

        if os.path.exists(files['pos']) and os.path.getsize(files['pos']) > 0:
            pos = self._read_csv_safe(files['pos'], encoding='gbk')
            if pos is not None:
                pos = pos[~pos['持仓合约'].apply(lambda x: '合计' in x)]
                if not pos.empty:
                    pos['user'] = 'quant'
                    pos = pos[['持仓合约', '买卖', '实收保证金', '总仓', '持仓市值', 'user']].rename(columns={
                        '持仓合约': 'code',
                        '买卖': 'Direction',
                        '实收保证金': 'UsedMargin',
                        '总仓': 'vol',
                        '持仓市值': 'amount',
                    })
                    pos['Direction'] = pos['Direction'].apply(lambda x: -1 if '卖' in x else 1)
                    pos['amount'] = pos['amount'] * pos['Direction']
                    self._save_dataframe(pos, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_pos.csv"))

        if os.path.exists(files['assets']) and os.path.getsize(files['assets']) > 0:
            assets = self._read_csv_safe(files['assets'], encoding='gbk')
            if assets is not None:
                assets['date'] = self.date
                assets['Deposit'] = 0
                assets['Withdraw'] = 0
                assets = assets[['date', '动态权益', '占用保证金', '可用资金', 'Deposit', 'Withdraw']].rename(columns={
                    '动态权益': 'DynamicRights',
                    '占用保证金': 'Margin',
                    '可用资金': 'Available',
                    })
                self._save_dataframe(assets, os.path.join(fut_data_dir, f"{self.date}_{product}_fut_assets.csv"))

    def standard_fut_data(self):
        """标准化期货数据（无限易和手动）"""
        fut_auto_dir = "E:/future_auto"
        fut_order, fut_pos, fut_assets = [], [], []

        # 收集所有无限易数据
        for client_dir in os.listdir(fut_auto_dir):
            client_full = os.path.join(fut_auto_dir, client_dir)
            if not os.path.isdir(client_full):
                continue
            date_folders = match_folders(client_full, self.date)
            if not date_folders:
                continue
            folder = os.path.join(client_full, date_folders[0], "dump")
            order_file = os.path.join(folder, "Order.csv")
            if os.path.exists(order_file):
                df = self._read_csv_safe(order_file, encoding='latin1')
                if df is not None:
                    fut_order.append(df)
            pos_file = os.path.join(folder, "PositionDetail.csv")
            if os.path.exists(pos_file):
                df = self._read_csv_safe(pos_file, encoding='latin1')
                if df is not None:
                    fut_pos.append(df)
            assets_file = os.path.join(folder, "InvestorAccount.csv")
            if os.path.exists(assets_file):
                df = self._read_csv_safe(assets_file, encoding='latin1')
                if df is not None:
                    fut_assets.append(df)

        fut_order = pd.concat(fut_order, ignore_index=True) if fut_order else pd.DataFrame()
        fut_pos = pd.concat(fut_pos, ignore_index=True) if fut_pos else pd.DataFrame()
        fut_assets = pd.concat(fut_assets, ignore_index=True) if fut_assets else pd.DataFrame()
        # print(fut_order)

        # 按产品处理
        for product in self.product_info:
            if product == 'PBZS1H11':
                self._process_hand_futures(product)
            else:
                self._process_futures_common(fut_order, fut_pos, fut_assets, product)

    def standard_option_data(self):
        """标准化期权数据"""
        opt_dir = "E:/期权数据"
        opt_data_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/opt_data", self.date)
        os.makedirs(opt_data_dir, exist_ok=True)

        for product in self.product_info:
            option_info = self.product_info.get(product, {}).get('option_info')
            if not option_info:
                continue

            opt_order, opt_pos, opt_assets = [], [], []
            for acct in option_info.values():
                cn_date = f"{self.date[:4]}年{int(self.date[4:6])}月{int(self.date[6:8])}日"
                order_file = os.path.join(opt_dir, acct, f"{cn_date}期权委托报表.csv")
                if os.path.exists(order_file):
                    df = self._read_csv_safe(order_file, encoding='gbk')
                    if df is not None:
                        opt_order.append(df)
                pos_file = os.path.join(opt_dir, acct, f"{cn_date}期权持仓报表.csv")
                if os.path.exists(pos_file):
                    df = self._read_csv_safe(pos_file, encoding='gbk')
                    if df is not None:
                        opt_pos.append(df)
                assets_file = os.path.join(opt_dir, acct, f"{cn_date}期权资金明细.csv")
                if os.path.exists(assets_file):
                    df = self._read_csv_safe(assets_file, encoding='gbk')
                    if df is not None:
                        opt_assets.append(df)

            if opt_order:
                self._save_dataframe(pd.concat(opt_order, ignore_index=True),
                                     os.path.join(opt_data_dir, f"{self.date}_{product}_opt_order.csv"))
            if opt_pos:
                self._save_dataframe(pd.concat(opt_pos, ignore_index=True),
                                     os.path.join(opt_data_dir, f"{self.date}_{product}_opt_pos.csv"))
            if opt_assets:
                self._save_dataframe(pd.concat(opt_assets, ignore_index=True),
                                     os.path.join(opt_data_dir, f"{self.date}_{product}_opt_assets.csv"))

    def _standardize_entrust(self, entrust, product, acct):
        """标准化单个账户的委托记录"""
        product_info = self.product_info[product]
        acct_info = product_info['acct_info'][acct]

        if acct == 'zhaoshang_PBTZ2H':
            # 从原始招商数据读取
            raw_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_other_data", self.date)
            file_path = os.path.join(raw_path, f"{self.date}_raw_zhaoshang_PBTZ2H_entrust.csv")
            if not os.path.exists(file_path):
                return None
            data = pd.read_csv(file_path)
            data['vendor'] = data['algo'].apply(lambda x: acct_info['adj_algo'] if x == 'adj' else acct_info['t0_algo'])
            data['time_period'] = data['algo'].apply(
                lambda x: f"{acct_info['adj_algo_starttime']}-{acct_info['adj_algo_endtime']}" if x == 'adj'
                else f"{acct_info['t0_algo_starttime']}-{acct_info['t0_algo_endtime']}")
        else:
            acct_id = acct_info['acct_num']
            data = entrust[entrust['资金账号'].astype(str).isin([str(acct_id)])].copy()
            if data.empty:
                return None
            data = data[~data['证券代码'].astype(str).str.contains('申购|债|新股|标准券|回购|GC001', na=False)]
            # 识别 T0 委托
            all_nums = set()
            # 从 raw_t0_data 读取
            t0_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_t0_data")
            file1 = match_file_v2(t0_dir, acct_id, self._date)
            if file1:
                try:
                    t0_data = pd.read_csv(file1, encoding='gbk')
                    t0_data['委托编号'] = t0_data['委托编号'].apply(
                        lambda x: int(str(x).split('||')[1].split('|')[0]) if len(str(x)) >= 10 else x)
                    all_nums.update(t0_data['委托编号'].tolist())
                except Exception:
                    pass
            # 从道和方舟读取
            file2 = match_file_v2("E:/道和方舟导出", f"结算子单查询{acct_id}", self.date)
            if file2:
                try:
                    t0_data = pd.read_csv(file2)
                    all_nums.update(t0_data['OrderID'].tolist())
                except Exception:
                    pass
            # data['algo'] = data['合同编号'].apply(lambda x: 't0' if x in all_nums else 'adj')
            # 前两种方式都没有，那就按照时间来分类
            data['algo'] = data['委托时间'].apply(lambda x: 't0' if str(x) > '09:35:00' else 'adj')

            data['vendor'] = data['algo'].apply(lambda x: acct_info['adj_algo'] if x == 'adj' else acct_info['t0_algo'])
            print(data)
            data['time_period'] = data['algo'].apply(
                lambda x: f"{acct_info['adj_algo_starttime']}-{acct_info['adj_algo_endtime']}" if x == 'adj'
                else f"{acct_info['t0_algo_starttime']}-{acct_info['t0_algo_endtime']}")
            data = data.drop(columns=['资金账号', '报单来源', '下单方式'], errors='ignore')
            data = data.rename(columns={
                '委托日期': 'date', '证券代码': 'code', '买卖标记': 'dir', '委托类别': 'order_type',
                '委托时间': 'order_time', '委托价格': 'order_price', '委托量': 'order_vol',
                '委托状态': 'order_status', '成交数量': 'filled_vol', '成交均价': 'filled_price'
            })
            data['dir'] = data['dir'].apply(lambda x: 1 if str(x).endswith('买入') else -1)

        return data[['date', 'code', 'dir', 'order_type', 'order_time', 'order_price', 'order_vol',
                     'order_status', 'filled_vol', 'filled_price', 'algo', 'vendor', 'time_period']]

    def standard_entrust(self):
        """标准化所有账户的委托记录"""
        entrust_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_qmt_data", self.date, f"{self.date}_raw_qmt_entrust.csv")
        if not os.path.exists(entrust_path):
            return
        entrust = pd.read_csv(entrust_path)

        standard_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/standard_data/entrust", self.date)
        os.makedirs(standard_dir, exist_ok=True)

        for product, pinfo in self.product_info.items():
            for acct in pinfo.get('acct_info', {}):
                data = self._standardize_entrust(entrust, product, acct)
                if data is None or data.empty:
                    continue
                local_file = os.path.join(standard_dir, f"{self.date}_{acct}_entrust.csv")
                data.to_csv(local_file, index=False)
                remote_path = f"{self.standard_path}/entrust/{self.date}/{self.date}_{acct}_entrust.csv"
                self.transfer.upload_file(local_file, remote_path)

    def standard_pos(self):
        """标准化持仓信息"""
        pos_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_qmt_data", self.date, f"{self.date}_raw_qmt_pos.csv")
        if not os.path.exists(pos_path):
            return
        pos = pd.read_csv(pos_path)

        standard_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/standard_data/pos", self.date)
        os.makedirs(standard_dir, exist_ok=True)

        for product, pinfo in self.product_info.items():
            for acct in pinfo.get('acct_info', {}):
                if acct == 'zhaoshang_PBTZ2H':
                    raw_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_other_data", self.date)
                    file_path = os.path.join(raw_path, f"{self.date}_raw_zhaoshang_PBTZ2H_pos.csv")
                    if not os.path.exists(file_path):
                        continue
                    data = pd.read_csv(file_path)
                    data = data[~data['交易对代码'].apply(lambda x: 'IF' in str(x))].reset_index(drop=True)
                    data['交易对代码'] = data['交易对代码'].apply(lambda x: int(str(x[:6])))
                    data = data[['交易对代码', '持仓数量']].rename(columns={'交易对代码': 'code', '持仓数量': 'hold'})
                    data = data[data['hold'] > 0].reset_index(drop=True)
                else:
                    acct_id = pinfo['acct_info'][acct]['acct_num']
                    data = pos[pos['资金账号'].astype(str).isin([str(acct_id)])].copy()
                    if data.empty:
                        continue
                    # 过滤不需要的品种
                    data = data[~data['证券名称'].astype(str).str.contains('申购|债|新股|标准券|回购|新增|新标准券', na=False)]
                    data = data[data['当前拥股'].astype(float) > 0].reset_index(drop=True)
                    data = data[['证券代码', '当前拥股']].rename(columns={'证券代码': 'code', '当前拥股': 'hold'})
                if data.empty:
                    continue
                local_file = os.path.join(standard_dir, f"{self.date}_{acct}_pos.csv")
                data.to_csv(local_file, index=False)
                remote_path = f"{self.standard_path}/pos/{self.date}/{self.date}_{acct}_pos.csv"
                self.transfer.upload_file(local_file, remote_path)

    def standard_assets(self):
        """标准化资金信息"""
        assets_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_qmt_data", self.date, f"{self.date}_raw_qmt_assets.csv")
        if not os.path.exists(assets_path):
            return
        assets = pd.read_csv(assets_path)
        assets = assets[assets['账号启用'] == '是'].reset_index(drop=True)

        standard_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/standard_data/assets", self.date)
        os.makedirs(standard_dir, exist_ok=True)

        for product, pinfo in self.product_info.items():
            for acct in pinfo.get('acct_info', {}):
                if acct == 'zhaoshang_PBTZ2H':
                    raw_path = os.path.join(ROOT_PATH, "data/standard_trade_data/raw_other_data", self.date)
                    file_path = os.path.join(raw_path, f"{self.date}_raw_zhaoshang_PBTZ2H_assets.csv")
                    if not os.path.exists(file_path):
                        continue
                    data = pd.read_csv(file_path)
                else:
                    acct_id = pinfo['acct_info'][acct]['acct_num']
                    data = assets[assets['资金账号'].astype(str).isin([str(acct_id)])].copy()
                    if data.empty:
                        continue
                    data = data.fillna(0).rename(columns={
                        '资金账号': 'acct_id', '净资产': 'net_assets', '总资产': 'tot_assets',
                        '总负债': 'tot_debat', '可用金额': 'ava_cash', '交易日': 'date',
                        '股票总市值': 'stk_amount', '债券总市值': 'bond_amount', '基金总市值': 'fund_amount',
                        '回购总市值': 'reop_amount', '可买担保品资金': 'coll_cash',
                        '融资市值': 'rongzi_amount', '融券市值': 'rongquan_amount', '融券卖出资金': 'rqmc_amount'
                    })
                    data = data[['date', 'acct_id', 'net_assets', 'tot_assets', 'tot_debat', 'ava_cash', 'stk_amount',
                                 'bond_amount', 'fund_amount', 'reop_amount', 'coll_cash', 'rongzi_amount', 'rongquan_amount', 'rqmc_amount']]
                local_file = os.path.join(standard_dir, f"{self.date}_{acct}_assets.csv")
                data.to_csv(local_file, index=False)
                remote_path = f"{self.standard_path}/assets/{self.date}/{self.date}_{acct}_assets.csv"
                self.transfer.upload_file(local_file, remote_path)

    def upload_derived_data(self):
        """上传期货和期权数据（复用连接）"""
        fut_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/fut_data", self.date)
        remote_fut_dir = f"{self.standard_path}/fut_data/{self.date}"
        if os.path.exists(fut_dir):
            for file in os.listdir(fut_dir):
                full_path = os.path.join(fut_dir, file)
                self.transfer.upload_file(full_path, f"{remote_fut_dir}/{file}")

        opt_dir = os.path.join(ROOT_PATH, "data/standard_trade_data/opt_data", self.date)
        remote_opt_dir = f"{self.standard_path}/opt_data/{self.date}"
        if os.path.exists(opt_dir):
            for file in os.listdir(opt_dir):
                full_path = os.path.join(opt_dir, file)
                self.transfer.upload_file(full_path, f"{remote_opt_dir}/{file}")

    def main(self):
        """主流程"""
        # 创建远程目录（一次性）
        self.transfer.connect()
        remote_base = self.standard_path
        dirs = ['pos', 'entrust', 'assets', 'fut_data', 'opt_data']
        for d in dirs:
            self.transfer.mkdir_remote_folder(f"{remote_base}/{d}/{self.date}")

        # 处理数据
        self.get_qmt_data()
        self.get_zhaoshangDMA_data()
        logger.info("标准化委托记录")
        self.standard_entrust()
        logger.info("标准化持仓信息")
        self.standard_pos()
        logger.info("标准化资金信息")
        self.standard_assets()
        logger.info("标准化期货数据")
        self.standard_fut_data()
        logger.info("标准化期权数据")
        self.standard_option_data()
        logger.info("上传期货期权数据")
        self.upload_derived_data()

        # 所有任务完成后关闭连接
        self.transfer.disconnect()
        logger.info("所有任务完成")

if __name__ == '__main__':
    date = dt.datetime.now().strftime('%Y%m%d')
    # date = '20260309'
    STD = StandardTradeData(date)
    STD.main()

    # start_date = '20260309'
    # end_date = '20260402'
    # rk_dates = rqdatac.get_trading_dates(
    #     start_date=start_date, end_date=end_date)
    # print(rk_dates)
    # for _date in rk_dates:
    #     _date = str(_date).replace('-', '')
    #     print(_date)
    #     STD = StandardTradeData(_date)
    #     STD.main()