from email import header
import os
import pandas as pd
import numpy as np
import datetime as dt


# 格式化股票代码
def format_code(code):
    '''
    in: 1,000001,000001.XSHG
    out: 000001.SZ
    '''
    code = int(str(code)[:6])
    code = str(code).zfill(6)
    code = code + '.SH' if code.startswith('6') else code + '.SZ'
    return code 


# 格式化股票代码
def format_code_str(code):
    '''
    in: 1,000001,000001.XSHG
    out: 000001
    '''
    code = int(str(code)[:6])
    code = str(code).zfill(6)
    return code

# 招商DMA股票代码格式
def format_code_zhaoshangdma(code):
    '''
    001213.XSHE@CNY
    600332.XSHG@CNY

    '''
    code = int(str(code)[:6])
    code = str(code).zfill(6)
    code = code + '.XSHG@CNY' if code.startswith('6') else code + '.XSHE@CNY'
    return code

# EMSX代码格式
def format_code_EMSX(code):
    '''
    300308 CH Equity
    601988 CH Equity

    '''
    code = int(str(code)[:6])
    code = str(code).zfill(6)
    
    return f'{code} CH Equity'


data_path = r'\\192.168.3.100\samba'
haitongcredit_rzmr_path = os.path.join(data_path, 'data', 'raw', 'rzrq', 'haitongcredit_rzmr.csv')
haitongcredit_rzmr_df = pd.read_csv(haitongcredit_rzmr_path, encoding='gbk')
haitongcredit_rzmr_df = haitongcredit_rzmr_df[haitongcredit_rzmr_df['融资状态'] == '正常'].reset_index(drop=True)
haitongcredit_rzmr_codes = haitongcredit_rzmr_df['证券代码'].unique().tolist()

haitongcredit_danbaopin_df = pd.read_csv(r'\\192.168.3.100\samba\data\raw\rzrq\haitongcredit_danbaopin.csv', encoding='gbk')
haitongcredit_danbaopin_df = haitongcredit_danbaopin_df[haitongcredit_danbaopin_df['是否可做担保'] == '正常'].reset_index(drop=True)
haitongcredit_danbaopin_codes = haitongcredit_danbaopin_df['证券代码'].unique().tolist()


# 调仓 20250718_ATX_gtja_PBPFZX1H_adj_KF_0930_0940.csv
def format_adj_algo(mo_path, data, date, adj_client, adj_acct, adj_algo, acct, adj_algo_starttime, adj_algo_endtime, rzrq=False, algo_type='adj'):
    '''
    data: 'code', 'reduce_vol1', 'close', 'adj_value'
    '''

    mo_path = f'{mo_path}/{date}'
    if not os.path.exists(mo_path):
        os.makedirs(mo_path)

    if adj_client == 'ATX':
        data['算法类型'] = 'VWAP'
        data['账户名称'] = adj_acct
        data['算法实例'] = 'kf_vwap_plus' if adj_algo == 'KF' else 'HX_SMART_VWAP' if adj_algo == 'HX' else 'ft_vwap_ai' if adj_algo == 'FT' else 'HX_SMART_VWAP' if adj_algo == 'HX_5min' else ''
        data['证券代码'] = data['code'].apply(format_code)
        data['任务数量'] = data['adj1'].astype(int).abs()
        if not rzrq:
            data['交易方向'] = np.where(data['adj1'] >= 0, '买入', '卖出')
            # data['交易方向'] = np.where(data['交易方向'] == '卖出', data['交易方向'], np.where(data['证券代码'].isin(haitongcredit_rzmr_codes), '融资买入', data['交易方向']))
        else:
            data['交易方向'] = np.where(data['adj1'] < 0, '卖出', np.where(data['code'].isin(haitongcredit_rzmr_codes), '融资买入', '买入'))
        data['开始时间'] = f'{date}T{adj_algo_starttime}000'
        data['结束时间'] = f'{date}T{adj_algo_endtime}000'
        data['涨跌停是否继续执行'] = '涨停不卖跌停不买'
        data['过期后是否继续执行'] = '否'
        # data['其他参数'] = f"篮子编号=adj_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
        data['其他参数'] = np.where(adj_algo == 'HX_5min', f"篮子编号=adj_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}:otherParams=remark&&hx_5min_trading", f"篮子编号=adj_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}")
        data['交易市场'] = ''

        data = data[['算法类型', '账户名称', '算法实例', '证券代码', '任务数量', '交易方向', '开始时间', '结束时间', '涨跌停是否继续执行', '过期后是否继续执行', '其他参数', '交易市场']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{adj_client}_{algo_type}_{adj_algo}_{adj_algo_starttime[:4]}_{adj_algo_endtime[:4]}_algo.csv', encoding='gbk', index=False)
    
    elif adj_client == '交易大师':
        data['资金账号'] = adj_acct
        data['标的代码'] = data['code'].apply(format_code_zhaoshangdma)
        data['申请数量'] = data['adj1'].astype(int).abs()
        data['买卖方向'] = np.where(data['adj1'] >= 0, '买入开仓', '卖出平仓')
        data['起始时间'] = f'{adj_algo_starttime[:2]}:{adj_algo_starttime[2:4]}:{adj_algo_starttime[4:6]}'
        data['截止时间'] = f'{adj_algo_endtime[:2]}:{adj_algo_endtime[2:4]}:{adj_algo_endtime[4:6]}'
        data['算法类型'] = 'kf_vwap_plus' if adj_algo == 'KF' else 'HX_SMART_VWAP' if adj_algo == 'HX' else 'ft_vwap_ai' if adj_algo == 'FT' else ''
        data['算法参数'] = ''
        data['节点（多节点时必填）'] = ''
        data = data[['资金账号', '标的代码', '申请数量', '买卖方向', '起始时间', '截止时间', '算法类型', '算法参数', '节点（多节点时必填）']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{adj_client}_{algo_type}_{adj_algo}_{adj_algo_starttime[:4]}_{adj_algo_endtime[:4]}_algo.csv', index=False)
        
    elif adj_client == 'bloomberg':
        data['broker'] = adj_acct
        data['ticker'] = data['code'].apply(format_code_EMSX)
        data['amount'] = data['adj1'].astype(int).abs()
        data['side'] = np.where(data['adj1'] >= 0, 'BUY', 'SELL')
        data['start_time'] = f'{adj_algo_starttime[:2]}:{adj_algo_starttime[2:4]}:{adj_algo_starttime[4:6]}'
        data['end_time'] = f'{adj_algo_endtime[:2]}:{adj_algo_endtime[2:4]}:{adj_algo_endtime[4:6]}'
        data['strategy'] = 'kf_vwap_plus' if adj_algo == 'KF' else 'HX_SMART_VWAP' if adj_algo == 'HX' else 'ft_vwap_ai' if adj_algo == 'FT' else 'VWAP' if adj_algo == 'EMSX' else ''
        data['hand_instruction'] = 'MAN'
        data['order_type'] = 'MKT'
        data['tif'] = 'DAY'
        data = data[['amount', 'broker', 'side', 'hand_instruction', 'order_type', 'ticker', 'tif', 'strategy', 'start_time', 'end_time']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{adj_client}_{algo_type}_{adj_algo}_{adj_algo_starttime[:4]}_{adj_algo_endtime[:4]}_algo.csv', index=False)

    elif adj_client == 'CATS':
        data['acct_type'] = 'FEQD'
        data['acct'] = adj_acct
        data['ticker'] = data['code'].apply(format_code)
        data['amount'] = data['adj1'].astype(int).abs()
        data['side'] = np.where(data['adj1'] >= 0, '1', '2')
        data['strategy'] = 'kf_vwap_plus' if adj_algo == 'KF' else 'HX_SMART_VWAP' if adj_algo == 'HX' else 'ft_vwap_ai' if adj_algo == 'FT' else 'VWAP' if adj_algo == 'EMSX' else 'VWAP3' if adj_algo == 'VWAP3' else ''
        data['params'] = f'beginTime={adj_algo_starttime};endTime={adj_algo_endtime};limitPrice=0;participateRate=0;tradingStyle=1'

        data = data[['amount', 'broker', 'side', 'ticker', 'strategy', 'start_time', 'end_time']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{adj_client}_{algo_type}_{adj_algo}_{adj_algo_starttime[:4]}_{adj_algo_endtime[:4]}_algo.csv', index=False, header=None)
        

# T0 20250718_ATX_gtja_PBPFZX1H_t0_KF_0930_0940.csv
def format_t0_algo(mo_path, data, date, t0_client, t0_acct, t0_algo, acct, t0_algo_starttime, t0_algo_endtime, rzrq=False):
    '''
    'code', 'left_vol', 'close'
    '''
    mo_path = f'{mo_path}/{date}'
    if not os.path.exists(mo_path):
        os.makedirs(mo_path)

    if t0_client == 'ATX':
        data['算法类型'] = 'T0'
        data['账户名称'] = t0_acct
        data['算法实例'] = 'kf_t0' if t0_algo == 'KF' else 'HX_SMART_T0' if t0_algo == 'HX' else 'ft_t0' if t0_algo == 'FT' else ''
        data['证券代码'] = data['code'].apply(format_code)
        data['任务数量'] = data['left_vol'].astype(int)
        data['买入方向'] = '买入'
        data['卖出方向'] = '卖出'
        data['开始时间'] = f'{date}T{t0_algo_starttime}000'
        data['结束时间'] = f'{date}T{t0_algo_endtime}000'
        data['涨跌停是否继续执行'] = '涨停不卖跌停不买'
        data['过期后是否继续执行'] = '否'
        data['其他参数'] = f"篮子编号=t0_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
        data['交易市场'] = ''
        data = data[['算法类型', '账户名称', '算法实例', '证券代码', '任务数量', '买入方向', '卖出方向', '开始时间', '结束时间', '涨跌停是否继续执行', '过期后是否继续执行', '其他参数', '交易市场']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{t0_client}_t0_{t0_algo}_{t0_algo_starttime[:4]}_{t0_algo_endtime[:4]}_algo.csv', index=False)
    
    elif t0_client == '交易大师':
        data['资金账号'] = t0_acct
        data['标的代码'] = data['code'].apply(format_code_zhaoshangdma)
        data['申请数量'] = data['left_vol'].astype(int).abs()
        data['买入方向'] = '买入开仓'
        data['卖出方向'] = '卖出平仓'
        data['起始时间'] = f'{t0_algo_starttime[:2]}:{t0_algo_starttime[2:4]}:{t0_algo_starttime[4:6]}'
        data['截止时间'] = f'{t0_algo_endtime[:2]}:{t0_algo_endtime[2:4]}:{t0_algo_endtime[4:6]}'
        data['算法类型'] = 'kf_t0' if t0_algo == 'KF' else 'HX_SMART_T0' if t0_algo == 'HX' else 'ft_t0' if t0_algo == 'FT' else 'YR_T0'
        data['算法参数'] = ''
        data['节点（多节点时必填）'] = ''
        data = data[['资金账号', '标的代码', '申请数量', '买入方向', '卖出方向', '起始时间', '截止时间', '算法类型', '算法参数', '节点（多节点时必填）']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{t0_client}_t0_{t0_algo}_{t0_algo_starttime[:4]}_{t0_algo_endtime[:4]}_algo.csv', index=False)
            
    elif t0_client == '道和方舟_YR':
        # code	volume	buyCredit	sellCredit
        data['code'] = data['code'].apply(format_code_str)
        data['volume'] = data['left_vol'].astype(int).abs()
        data['buyCredit'] = 0
        data['sellCredit'] = 0
        data = data[['code', 'volume', 'buyCredit', 'sellCredit']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{t0_client}_t0_{t0_algo}_{t0_algo_starttime[:4]}_{t0_algo_endtime[:4]}_algo.csv', index=False)  
    
    elif t0_client == '道和方舟':
        data['代码'] = data['code'].apply(format_code_str)
        data['样本数量'] = data['left_vol'].astype(int).abs()
        data['权重'] = 0
        data['买委托方式'] = '买入'
        data['卖委托方式'] = '卖出'
        data = data[['代码', '样本数量', '权重', '买委托方式', '卖委托方式']]
        data.to_csv(f'{mo_path}/{date}_{acct}_{t0_client}_t0_{t0_algo}_{t0_algo_starttime[:4]}_{t0_algo_endtime[:4]}_algo.csv', encoding='gbk', index=False)
            



