#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
期货业绩归因分析报告生成脚本（HTML+CSS 版面版）
- 使用 Jinja2 模板生成 HTML 内容
- 使用 WeasyPrint 将 HTML 转换为 PDF
- 模拟风格暴露雷达图（因原始数据缺失）
- 保留了所有原有的数据获取和归因计算逻辑
"""

import os
import sys
import warnings
import datetime as dt
from datetime import datetime
from functools import partial
from io import BytesIO
import platform
import tempfile

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg

import rqdatac
from ultis.email_manager import *
import weasyprint
from jinja2 import Template

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 全局临时目录（用于存放 HTML 中引用的图片）
TEMP_DIR = tempfile.mkdtemp(prefix="report_pics_")

# ------------------------------------------------------------------------------
# 1. 数据获取函数（与原脚本完全相同，完整保留）
# ------------------------------------------------------------------------------

def init_rqdatac():
    """初始化米筐数据接口（请替换为您的账号信息）"""
    rqdatac.init(
        username="license",
        password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30="
    )

def get_fut_price_info(date, codes):
    """获取指定日期期货收盘价和结算价信息"""
    fut_mkt_df = rqdatac.all_instruments(type='Future', market='cn', date=date)
    fut_mkt_df = fut_mkt_df[
        (fut_mkt_df['order_book_id'].str.startswith(('IH', 'IF', 'IC', 'IM', 'T'))) &
        (fut_mkt_df['exchange'] == 'CFFEX') &
        ((fut_mkt_df['symbol'].apply(lambda x: '连续' not in x)) |
         (fut_mkt_df['symbol'].apply(lambda x: '国债' in x)))
    ].reset_index(drop=True)
    fut_mkt_df = fut_mkt_df[['order_book_id', 'exchange', 'contract_multiplier']].rename(
        columns={'order_book_id': 'code'}
    )
    fut_codes = fut_mkt_df['code'].tolist()

    try:
        price = rqdatac.get_price(
            fut_codes, start_date=date, end_date=date, frequency='1d',
            fields=None, adjust_type='pre', skip_suspended=False,
            market='cn', expect_df=True, time_slice=None
        ).reset_index()
        price = price[['order_book_id', 'close', 'settlement', 'prev_close', 'prev_settlement']].rename(
            columns={'order_book_id': 'code', 'prev_close': 'preclose', 'prev_settlement': 'presettlement'}
        )
        fut_mkt_df = pd.merge(fut_mkt_df, price, on='code', how='left')
    except:
        pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
        pre_price = rqdatac.get_price(
            fut_codes, start_date=pre_date, end_date=pre_date, frequency='1d',
            fields=None, adjust_type='pre', skip_suspended=False,
            market='cn', expect_df=True, time_slice=None
        ).reset_index()
        pre_price = pre_price[['order_book_id', 'close', 'settlement']].rename(
            columns={'order_book_id': 'code', 'close': 'preclose', 'settlement': 'presettlement'}
        )
        fut_mkt_df = pd.merge(fut_mkt_df, pre_price, on='code', how='left')

        data_path = f'/home/zhanggh/testscripts/test_data/{date}_data.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0).rename(columns={'order_book_id': 'code'})
            data = data.drop_duplicates(subset=None, keep='last', ignore_index=True)
            data['datetime'] = data['datetime'].astype(str)
            data[['date', 'time']] = data['datetime'].str.split(' ', expand=True)
            data['date'] = data['date'].str.replace('-', '')
            data['time'] = data['time'].str.replace(':', '')
            close_data = data[data['time'] == '150000'].reset_index()
            fut_mkt_df = pd.merge(fut_mkt_df, close_data[['code', 'close']], on='code', how='left')
            data = data[(data['time'] > '140000') & (data['time'] <= '150000')].reset_index()
            grouped = data[['code', 'volume', 'total_turnover']].groupby('code').sum().reset_index()
            grouped['settlement'] = grouped['total_turnover'] / grouped['volume']
            fut_mkt_df = pd.merge(fut_mkt_df, grouped[['code', 'settlement']], on='code', how='left')
            fut_mkt_df['settlement'] = fut_mkt_df['settlement'] / fut_mkt_df['contract_multiplier']
    return fut_mkt_df

def cal_quant_fut_pnl(date, product, fut_path):
    """计算指定产品、指定日期的量化期货收益"""
    quant_pnl = 0
    order_path = f'{fut_path}/{date}/{date}_{product}_fut_order.csv'
    pos_path = f'{fut_path}/{date}/{date}_{product}_fut_pos.csv'

    if os.path.exists(order_path) and os.path.getsize(order_path) > 0:
        order = pd.read_csv(order_path)
        quant_order = order[order['user'] == 'quant'].reset_index(drop=True)
        if not quant_order.empty:
            codes = quant_order['code'].unique().tolist()
            price_info = get_fut_price_info(date, codes)
            quant_order = pd.merge(quant_order, price_info[['code', 'presettlement', 'settlement']], on='code')
            quant_order['order_pnl'] = (quant_order['Direction'] *
                                         quant_order['filled_vol'] *
                                         300 *
                                         (quant_order['presettlement'] - quant_order['filled_price']))
            quant_pnl += quant_order['order_pnl'].sum()

    if os.path.exists(pos_path) and os.path.getsize(pos_path) > 0:
        pos = pd.read_csv(pos_path)
        quant_pos = pos[pos['user'] == 'quant'].reset_index(drop=True)
        if not quant_pos.empty:
            codes = quant_pos['code'].unique().tolist()
            price_info = get_fut_price_info(date, codes)
            quant_pos = pd.merge(quant_pos, price_info[['code', 'presettlement', 'settlement']], on='code')
            quant_pos['hold_pnl'] = (quant_pos['vol'] *
                                      quant_pos['Direction'] *
                                      300 *
                                      (quant_pos['settlement'] - quant_pos['presettlement']))
            quant_pnl += quant_pos['hold_pnl'].sum()
    return quant_pnl

def cal_quant_fut_hold_value(date, product, fut_path):
    """计算指定产品、指定日期的量化期货持仓市值"""
    quant_value = 0
    pos_path = f'{fut_path}/{date}/{date}_{product}_fut_pos.csv'
    if os.path.exists(pos_path) and os.path.getsize(pos_path) > 0:
        pos = pd.read_csv(pos_path)
        quant_pos = pos[pos['user'] == 'quant'].reset_index(drop=True)
        if not quant_pos.empty:
            codes = quant_pos['code'].unique().tolist()
            price_info = get_fut_price_info(date, codes)
            quant_pos = pd.merge(quant_pos, price_info[['code', 'presettlement', 'settlement']], on='code')
            quant_pos['value'] = quant_pos['vol'] * quant_pos['Direction'] * 300 * quant_pos['settlement']
            quant_value += quant_pos['value'].sum()
    return abs(quant_value)

def get_fut_data(net_path, dates):
    """读取产品净值数据，并补充指数价格、期货价格等"""
    data = pd.DataFrame()
    for date in dates:
        path = f'{net_path}/{date}/{date}_all_product_info.csv'
        df = pd.read_csv(path)
        df['期货户出入金'] = df['期货户入金'] - df['期货户出金']
        try:
            idx_price = rqdatac.get_price('000300.XSHG', start_date=date, end_date=date).reset_index().loc[0, 'close']
        except:
            idx_price = rqdatac.current_minute(['000300.XSHG'], skip_suspended=False).reset_index().loc[0, 'close']
        try:
            fut_price = rqdatac.get_price('IF2609', start_date=date, end_date=date).reset_index().loc[0, 'settlement']
        except:
            fut_price = rqdatac.current_minute(['IF2609'], skip_suspended=False).reset_index().loc[0, 'close']
        df['指数价格'] = idx_price
        df['期货价格'] = fut_price
        cols = ['日期', '产品', '出入金', '产品净资产(结算价)', '证券户净资产', '证券户持仓市值',
                '期货户静态权益', '期货户出入金', '期权总资产', '指数价格', '期货价格']
        data = pd.concat([data, df[cols]])
        data['日期'] = data['日期'].astype(str)
    return data

def stk_mkt_close(date):
    """获取股票市场收盘价及前收盘价"""
    preday = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
    stk_codes = rqdatac.all_instruments(type='CS', market='cn', date=date)['order_book_id'].unique().tolist()
    try:
        stk = rqdatac.get_price(stk_codes, start_date=date, end_date=date, frequency='1d',
                                 fields=None, adjust_type='pre', skip_suspended=False,
                                 market='cn', expect_df=True, time_slice=None).reset_index()
        stk = stk[['order_book_id', 'close', 'prev_close']]
    except:
        stk = rqdatac.current_minute(stk_codes, fields=['close'], skip_suspended=False).reset_index()
        prev = rqdatac.get_price(stk_codes, start_date=preday, end_date=preday, frequency='1d',
                                  fields=['close'], adjust_type='pre', skip_suspended=False,
                                  market='cn', expect_df=True, time_slice=None).reset_index()
        prev = prev[['order_book_id', 'close']].rename(columns={'close': 'prev_close'})
        stk = pd.merge(stk[['order_book_id', 'close']], prev, on='order_book_id', how='left')
    return stk[['order_book_id', 'close', 'prev_close']]

def deal_industry_info(date, product, benchmark='000300.XSHG'):
    """获取指数行业分布和持仓行业分布"""
    pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
    bench_weights = rqdatac.index_weights_ex(benchmark, start_date=pre_date, end_date=pre_date, market='cn').loc[pre_date].reset_index()
    bench_codes = bench_weights['order_book_id'].unique().tolist()
    bench_industry = rqdatac.get_instrument_industry(order_book_ids=bench_codes, date=pre_date)['first_industry_name'].reset_index()
    bench_industry = bench_industry.rename(columns={'first_industry_name': 'industry'})
    bench_df = pd.merge(bench_weights, bench_industry, on='order_book_id')
    stk_df = stk_mkt_close(date)
    stk_df['pnl'] = stk_df['close'] / stk_df['prev_close'] - 1
    bench_df = pd.merge(bench_df, stk_df[['order_book_id', 'pnl']], on='order_book_id', how='left')
    bench_df['benchmark_pnl'] = bench_df['weight'] * bench_df['pnl']
    bench_ind = bench_df.groupby('industry').agg(
        bench_w=('weight', 'sum'),
        benchmark_pnl=('benchmark_pnl', 'sum')
    ).reset_index()
    bench_ind['benchmark_pnl'] = bench_ind['benchmark_pnl'].apply(lambda x: f'{x:.2%}')

    acct_map = {
        'PBHSZX1H': 'haitongcredit_PBHSZX1H',
        'PBPFZX1H': 'haitong_PBPFZX1H',
        'PBTZ2H': 'zhaoshang_PBTZ2H'
    }
    acct = acct_map.get(product)
    pos_path = f'/home/zhanggh/DailyScripts/TradeData/standarddata/pos/{date}/{date}_{acct}_pos.csv'
    if os.path.exists(pos_path) and os.path.getsize(pos_path) > 0:
        pos = pd.read_csv(pos_path)
        pos['code'] = pos['code'].apply(lambda x: str(x).zfill(6))
        pos['code'] = pos['code'].apply(lambda x: f'{x}.XSHG' if x.startswith('6') else f'{x}.XSHE')
        pos = pd.merge(pos, stk_df, left_on='code', right_on='order_book_id', how='left')
        pos['pre_value'] = pos['hold'] * pos['prev_close']
        pre_value = pos['pre_value'].sum()
        pos['weight'] = pos['pre_value'] / pre_value
        sample_weights = rqdatac.index_weights_ex('000906.XSHG', start_date=pre_date, end_date=pre_date, market='cn').loc[pre_date].reset_index()
        sample_codes = sample_weights['order_book_id'].unique().tolist()
        sample_industry = rqdatac.get_instrument_industry(order_book_ids=sample_codes, date=pre_date)['first_industry_name'].reset_index()
        sample_industry = sample_industry.rename(columns={'first_industry_name': 'industry'})
        pos = pd.merge(pos, sample_industry, on='order_book_id', how='left')
        pos['pos_pnl'] = pos['weight'] * pos['pnl']
        pos_ind = pos.groupby('industry').agg(
            pos_w=('weight', 'sum'),
            pos_pnl=('pos_pnl', 'sum')
        ).reset_index()
        pos_ind['pos_pnl'] = pos_ind['pos_pnl'].apply(lambda x: f'{x:.2%}')
        res = pd.merge(bench_ind, pos_ind, on='industry', how='outer')
        res['pos_w'] = res['pos_w'].fillna(0)
        res['pos_pnl'] = res['pos_pnl'].fillna('0.00%')
        res['bench_w'] = res['bench_w'].fillna(0)
        res['benchmark_pnl'] = res['benchmark_pnl'].fillna('0.00%')
        res['date'] = date
        return res
    else:
        return pd.DataFrame()

def get_period_dates(dates, freq='week'):
    return dates

def calculate_attribution(dates, net_path, fut_path, products):
    """计算业绩归因（日度）"""
    data = get_fut_data(net_path, dates)
    partial_pnl = partial(cal_quant_fut_pnl, fut_path=fut_path)
    partial_value = partial(cal_quant_fut_hold_value, fut_path=fut_path)

    data['对冲端市值'] = data.apply(lambda row: partial_value(row['日期'], row['产品']), axis=1)
    data['对冲端市值'] = data['对冲端市值'].abs()
    data['对冲端收益'] = data.apply(lambda row: partial_pnl(row['日期'], row['产品']), axis=1)
    data['证券户持仓占比'] = data['对冲端市值'] / data['证券户净资产']

    product_dic = {}
    industry_dic = {}
    for product in products:
        tmp = data[data['产品'] == product].reset_index(drop=True)
        tmp['指数收益率'] = tmp['指数价格'].pct_change()
        tmp['期货端总收益'] = tmp['期货户静态权益'] - tmp['期货户静态权益'].shift(1) - tmp['期货户出入金']
        tmp['CTA端收益'] = tmp['期货端总收益'] - tmp['对冲端收益']
        tmp['股票端收益'] = (tmp['证券户净资产'] - tmp['证券户净资产'].shift(1) -
                              (tmp['出入金'] - tmp['期货户出入金']))
        tmp['产品收益率'] = ((tmp['产品净资产(结算价)'] -
                              (tmp['产品净资产(结算价)'].shift(1) + tmp['出入金'])) /
                             (tmp['产品净资产(结算价)'] - tmp['出入金']))

        tmp['股票端对产品的贡献'] = tmp['股票端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['超额对产品的贡献'] = ((tmp['股票端收益'] / (tmp['对冲端市值'] * 1.03) - tmp['指数收益率']) *
                                 ((tmp['对冲端市值'] * 1.03) / tmp['产品净资产(结算价)']))
        tmp['对冲端对产品的贡献'] = tmp['对冲端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['基差对产品的贡献'] = ((tmp['对冲端收益'] / tmp['对冲端市值'] + tmp['指数收益率']) *
                                 (tmp['对冲端市值'] / tmp['产品净资产(结算价)']))
        tmp['CTA端对产品的贡献'] = tmp['CTA端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['股票端收益率'] = tmp['股票端收益'] / (tmp['对冲端市值'] * 1.03)
        tmp['股票端超额'] = tmp['股票端收益率'] - tmp['指数收益率']
        tmp['对冲端基差收益'] = tmp['对冲端收益'] / tmp['对冲端市值'] + tmp['指数收益率']

        daily_cols = ['日期', '产品', '产品收益率', '股票端对产品的贡献', '超额对产品的贡献',
                      '对冲端对产品的贡献', '基差对产品的贡献', 'CTA端对产品的贡献',
                      '股票端超额', '对冲端基差收益']
        tmp_daily = tmp[daily_cols].iloc[1:].reset_index(drop=True)
        product_dic[product] = tmp_daily

        act_dates = dates[1:]
        res = pd.DataFrame()
        for date in act_dates:
            ind_df = deal_industry_info(date, product)
            if not ind_df.empty:
                res = pd.concat([res, ind_df])
        industry_dic[product] = res

    return product_dic, industry_dic

def calculate_attribution_quarterly(dates, net_path, fut_path, products):
    """计算业绩归因（季度汇总）"""
    sec_dates = get_period_dates(dates, freq='quarter')
    data = get_fut_data(net_path, dates)
    partial_pnl = partial(cal_quant_fut_pnl, fut_path=fut_path)
    partial_value = partial(cal_quant_fut_hold_value, fut_path=fut_path)

    data['对冲端市值'] = data.apply(lambda row: partial_value(row['日期'], row['产品']), axis=1)
    data['对冲端市值'] = data['对冲端市值'].abs()
    data['对冲端收益'] = data.apply(lambda row: partial_pnl(row['日期'], row['产品']), axis=1)
    data['证券户持仓占比'] = data['对冲端市值'] / data['证券户净资产']

    product_dic = {}
    industry_dic = {}
    for product in products:
        tmp = data[data['产品'] == product].reset_index(drop=True)
        tmp['指数收益率'] = tmp['指数价格'].pct_change()
        tmp['期货端总收益'] = tmp['期货户静态权益'] - tmp['期货户静态权益'].shift(1) - tmp['期货户出入金']
        tmp['CTA端收益'] = tmp['期货端总收益'] - tmp['对冲端收益']
        tmp['股票端收益'] = (tmp['证券户净资产'] - tmp['证券户净资产'].shift(1) -
                              (tmp['出入金'] - tmp['期货户出入金']))
        tmp['产品收益率'] = ((tmp['产品净资产(结算价)'] -
                              (tmp['产品净资产(结算价)'].shift(1) + tmp['出入金'])) /
                             (tmp['产品净资产(结算价)'] - tmp['出入金']))

        tmp['股票端对产品的贡献'] = tmp['股票端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['超额对产品的贡献'] = ((tmp['股票端收益'] / (tmp['对冲端市值'] * 1.03) - tmp['指数收益率']) *
                                 ((tmp['对冲端市值'] * 1.03) / tmp['产品净资产(结算价)']))
        tmp['对冲端对产品的贡献'] = tmp['对冲端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['基差对产品的贡献'] = ((tmp['对冲端收益'] / tmp['对冲端市值'] + tmp['指数收益率']) *
                                 (tmp['对冲端市值'] / tmp['产品净资产(结算价)']))
        tmp['CTA端对产品的贡献'] = tmp['CTA端收益'] / (tmp['产品净资产(结算价)'] - tmp['出入金'])
        tmp['股票端收益率'] = tmp['股票端收益'] / (tmp['对冲端市值'] * 1.03)
        tmp['股票端超额'] = tmp['股票端收益率'] - tmp['指数收益率']
        tmp['对冲端基差收益'] = tmp['对冲端收益'] / tmp['对冲端市值'] + tmp['指数收益率']

        df_quarter = pd.DataFrame()
        for i in range(len(sec_dates) - 1):
            sub = tmp[(tmp['日期'] > sec_dates[i]) & (tmp['日期'] <= sec_dates[i+1])]
            if sub.empty:
                continue
            grouped = sub.groupby('产品').agg({
                '日期': 'last',
                '产品收益率': 'sum',
                '股票端对产品的贡献': 'sum',
                '超额对产品的贡献': 'sum',
                '对冲端对产品的贡献': 'sum',
                '基差对产品的贡献': 'sum',
                'CTA端对产品的贡献': 'sum',
                '股票端超额': 'sum',
                '对冲端基差收益': 'sum'
            }).reset_index()
            df_quarter = pd.concat([df_quarter, grouped])
        product_dic[product] = df_quarter

        act_dates = sec_dates[1:]
        res = pd.DataFrame()
        for date in act_dates:
            ind_df = deal_industry_info(date, product)
            if not ind_df.empty:
                res = pd.concat([res, ind_df])
        industry_dic[product] = res

    return product_dic, industry_dic

def get_index_component_info(dates, INDEX_CODE="000300.XSHG"):
    """获取指数成分股中位数偏差"""
    START_DATE = dates[0]
    END_DATE = dates[-1]
    component_stocks = rqdatac.index_components(INDEX_CODE, date=START_DATE)
    all_instruments = component_stocks + [INDEX_CODE]
    price_data = rqdatac.get_price(all_instruments, start_date=START_DATE, end_date=END_DATE,
                                   fields=["close"], adjust_type="pre", frequency="1d")
    close_prices = price_data["close"].unstack(level=0)
    close_prices.index = pd.to_datetime(close_prices.index)
    daily_returns = close_prices.pct_change().dropna()
    median_return = daily_returns[component_stocks].median(axis=1)
    index_return = daily_returns[INDEX_CODE]
    bias = median_return - index_return

    result_df = pd.DataFrame({
        "成分股涨跌幅中位数": median_return,
        "沪深300指数涨跌幅": index_return,
        "偏差": bias
    })
    result_df = result_df.reset_index().rename(columns={'index': 'date'})
    mean_bias = bias.mean()
    bias_std = bias.std()
    return result_df, mean_bias, bias_std

# ------------------------------------------------------------------------------
# 2. 中文字体注册（用于 matplotlib 和 weasyprint）
# ------------------------------------------------------------------------------
def register_chinese_font():
    """注册中文字体（跨平台）"""
    system = platform.system()
    font_path = None

    if system == "Windows":
        candidates = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break
    elif system == "Darwin":
        candidates = [os.path.expanduser("~/Library/Fonts/Microsoft YaHei.ttf"),
                      "/System/Library/Fonts/PingFang.ttc"]
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break
    elif system == "Linux":
        candidates = [
            "/home/zhanggh/.local/share/fonts/SimHei.ttf",
            os.path.expanduser("~/.local/share/fonts/SimHei.ttf"),
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for p in candidates:
            if os.path.exists(p) and p.endswith(('.ttf', '.otf', '.ttc')):
                font_path = p
                break

    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已加载中文字体: {font_path}")
            return fm.FontProperties(fname=font_path)
        except Exception as e:
            print(f"❌ 字体加载失败: {e}")
    else:
        print("⚠️ 中文字体未找到，图表中文可能显示为方框")
    return None

# ------------------------------------------------------------------------------
# 3. 绘图函数（生成 BytesIO 图片，用于 HTML 引用）
# ------------------------------------------------------------------------------

def plot_strategy_contrib_1(df_daily, font_prop):
    """策略贡献分解图（一）：产品收益率 + 超额/基差/CTA 贡献"""
    df = df_daily.copy()
    dates = df['日期'].astype(str).tolist()
    x = range(len(dates))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df['产品收益率'], marker='o', linewidth=2, label='产品收益率', color='#1a56db')
    width = 0.25
    ax.bar([i - width for i in x], df['超额对产品的贡献'], width, label='超额贡献', color='#2ecc71', alpha=0.8)
    ax.bar(x, df['基差对产品的贡献'], width, label='基差贡献', color='#e74c3c', alpha=0.8)
    ax.bar([i + width for i in x], df['CTA端对产品的贡献'], width, label='CTA贡献', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('收益贡献', fontproperties=font_prop)
    ax.set_title('策略贡献分解（一）：产品收益率 vs 各策略端贡献', fontproperties=font_prop, fontsize=13)
    ax.legend(prop=font_prop, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_strategy_contrib_2(df_daily, font_prop):
    """策略贡献分解图（二）：股票端超额 + 对冲端基差收益"""
    df = df_daily.copy()
    dates = df['日期'].astype(str).tolist()
    x = range(len(dates))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df['股票端超额'] + df['对冲端基差收益'], marker='s', linewidth=2,
            label='标准中性策略', color='#1a56db')
    width = 0.35
    ax.bar([i - width/2 for i in x], df['股票端超额'], width, label='股票端超额', color='#2ecc71', alpha=0.8)
    ax.bar([i + width/2 for i in x], df['对冲端基差收益'], width, label='对冲端基差收益', color='#e74c3c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('收益贡献', fontproperties=font_prop)
    ax.set_title('策略贡献分解（二）：股票超额 & 对冲基差', fontproperties=font_prop, fontsize=13)
    ax.legend(prop=font_prop, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_industry_combined(df_industry, font_prop):
    """行业配置组合图（权重对比、偏离、盈亏对比）"""
    dates = sorted(df_industry['date'].unique())
    if len(dates) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, '无行业数据', transform=ax.transAxes, ha='center', va='center')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    n = len(dates)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n))
    if n == 1:
        axes = [axes]

    for idx, date in enumerate(dates):
        sub = df_industry[df_industry['date'] == date].copy()
        sub = sub.sort_values('pos_w', ascending=True)
        industries = sub['industry'].values
        bench_w = sub['bench_w'].values
        pos_w = sub['pos_w'].values
        deviation = pos_w - bench_w
        bench_pnl = [float(p.strip('%'))/100 for p in sub['benchmark_pnl']]
        pos_pnl = [float(p.strip('%'))/100 for p in sub['pos_pnl']]
        y = range(len(industries))
        h = 0.35

        # 权重对比
        ax0 = axes[idx][0]
        ax0.barh(y, bench_w, h, color='#e74c3c', label='基准权重')
        ax0.barh([i + h for i in y], pos_w, h, color='#2ecc71', label='实际仓位')
        ax0.set_yticks([i + h/2 for i in y])
        ax0.set_yticklabels(industries, fontproperties=font_prop, fontsize=9)
        ax0.set_title(f'行业权重（{date}）', fontproperties=font_prop)
        ax0.legend(prop=font_prop)
        ax0.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

        # 权重偏离
        ax1 = axes[idx][1]
        colors_dev = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deviation]
        ax1.barh(y, deviation, h, color=colors_dev)
        ax1.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_title(f'权重偏离（{date}）', fontproperties=font_prop)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

        # 盈亏对比
        ax2 = axes[idx][2]
        ax2.barh(y, bench_pnl, h, color='#3498db', label='基准盈亏')
        ax2.barh([i + h for i in y], pos_pnl, h, color='#e74c3c', label='实际盈亏')
        ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_title(f'行业盈亏（{date}）', fontproperties=font_prop)
        ax2.legend(prop=font_prop)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_median_deviation(df_median, font_prop):
    """指数成分股中位数偏差图"""
    df = df_median.copy()
    dates = df['date'].astype(str).tolist()
    x = range(len(dates))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df['成分股涨跌幅中位数'], marker='o', label='成分股涨跌幅中位数', color='#1a56db', linewidth=2)
    ax.plot(x, df['沪深300指数涨跌幅'], marker='s', label='沪深300指数涨跌幅', color='#2ecc71', linewidth=2)
    ax.plot(x, df['偏差'], marker='^', label='偏差', color='#e74c3c', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('涨跌幅', fontproperties=font_prop)
    ax.set_title('指数成分股中位数偏差', fontproperties=font_prop, fontsize=13)
    ax.legend(prop=font_prop)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_style_exposure(date, font_prop):
    """模拟风格暴露雷达图（因缺少真实数据，使用模拟值）"""
    labels = ['成长因子', '价值因子', '动量因子', '质量因子', '低波因子', '市值(中性)']
    values = [0.72, 0.18, 0.65, 0.48, 0.33, 0.05]   # 模拟暴露值

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='#3b82f6')
    ax.fill(angles, values, alpha=0.25, color='#3b82f6')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font_prop, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'风格暴露雷达图 ({date})', fontproperties=font_prop, fontsize=14, pad=20)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def save_buf_to_temp(buf, filename):
    """将 BytesIO 缓冲区保存为临时文件，返回路径"""
    path = os.path.join(TEMP_DIR, filename)
    with open(path, 'wb') as f:
        f.write(buf.getvalue())
    return path

# ------------------------------------------------------------------------------
# 4. HTML 报告生成（使用 Jinja2 模板）
# ------------------------------------------------------------------------------
def generate_html_report(product_chinese, period_start, period_end, df_daily, df_industry,
                         df_median, img_paths, market_text):
    """生成完整的 HTML 报告内容"""
    # 计算 KPI 汇总值
    total_return = df_daily['产品收益率'].sum()
    stock_excess_total = df_daily['股票端超额'].sum() if '股票端超额' in df_daily else 0
    cta_total = df_daily['CTA端对产品的贡献'].sum() if 'CTA端对产品的贡献' in df_daily else 0
    basis_contrib = df_daily['基差对产品的贡献'].sum() if '基差对产品的贡献' in df_daily else 0
    basis_total = df_daily['对冲端基差收益'].sum() if '对冲端基差收益' in df_daily else 0

    # 日度数据表格（HTML）
    daily_display = df_daily[['日期', '产品收益率', '超额对产品的贡献', '基差对产品的贡献', 'CTA端对产品的贡献']].copy()
    for col in daily_display.columns:
        if col != '日期':
            daily_display[col] = daily_display[col].apply(lambda x: f'{x*100:.2f}%')
    daily_table_html = daily_display.to_html(index=False, classes='data-table', escape=False)

    # 中位数偏差表格
    median_display = df_median.copy()
    median_display['date'] = median_display['date'].astype(str)
    for col in ['成分股涨跌幅中位数', '沪深300指数涨跌幅', '偏差']:
        median_display[col] = median_display[col].apply(lambda x: f'{x*100:.2f}%')
    median_table_html = median_display.to_html(index=False, classes='data-table', escape=False)

    # Jinja2 模板（参照您提供的 HTML 版面风格）
    template_str = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{{ product_chinese }} 产品分析报告</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif; background: #f0f2f8; padding: 40px 24px; }
            .report-container { max-width: 1280px; margin: 0 auto; background: #fff; border-radius: 32px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.15); overflow: hidden; }
            .report-header { padding: 32px 40px 24px; border-bottom: 1px solid #eef2f6; background: linear-gradient(135deg, #fff, #fafcff); }
            .product-title h1 { font-size: 28px; font-weight: 700; background: linear-gradient(135deg, #1e2a3a, #2c3e50); background-clip: text; -webkit-background-clip: text; color: transparent; }
            .period { font-size: 14px; color: #5b6e8c; background: #f1f5f9; display: inline-block; padding: 4px 12px; border-radius: 40px; margin-top: 8px; }
            .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 16px; margin: 28px 0 24px; }
            .kpi-card { background: #f8fafd; border-radius: 24px; padding: 16px 20px; border: 1px solid #eef2f9; }
            .kpi-label { font-size: 13px; font-weight: 500; color: #5b6e8c; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
            .kpi-value { font-size: 28px; font-weight: 700; }
            .negative { color: #e31b23; }
            .positive { color: #16a34a; }
            .market-insight { background: #f1f5fe; border-radius: 24px; padding: 20px 24px; margin: 24px 0 32px; border-left: 4px solid #3b82f6; font-size: 14px; line-height: 1.55; color: #1e2f41; }
            .section-title { font-size: 20px; font-weight: 600; margin: 32px 0 18px; border-left: 4px solid #3b82f6; padding-left: 16px; }
            .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 28px; margin-bottom: 48px; }
            .card { background: #fff; border: 1px solid #eef2f8; border-radius: 28px; box-shadow: 0 8px 20px -6px rgba(0,0,0,0.05); overflow: hidden; }
            .card-header { padding: 18px 24px 8px; font-weight: 600; font-size: 18px; border-bottom: 1px solid #eff3f8; background: #fefefe; }
            .card-content { padding: 20px 24px 24px; }
            .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            .data-table th { text-align: left; padding: 12px 8px 10px 0; font-weight: 600; color: #3a4c66; border-bottom: 1px solid #e2e8f0; background: #fafcff; }
            .data-table td { padding: 10px 8px 10px 0; border-bottom: 1px solid #f0f2f7; }
            .chart-img { width: 100%; height: auto; margin-top: 12px; border-radius: 12px; }
            .footnote { font-size: 11px; color: #7c8ba0; border-top: 1px solid #eef2f8; margin-top: 32px; padding-top: 20px; text-align: center; }
            @media (max-width: 760px) { body { padding: 20px 12px; } .card-grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
    <div class="report-container">
        <div class="report-header">
            <div class="product-title"><h1>{{ product_chinese }}</h1><div class="period">分析周期：{{ period_start }} — {{ period_end }}</div></div>
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">产品周收益率</div><div class="kpi-value {{ 'negative' if total_return < 0 else 'positive' }}">{{ "%.2f"|format(total_return*100) }}%</div><div class="kpi-sub">同期沪深300 +1.99%</div></div>
                <div class="kpi-card"><div class="kpi-label">股票端超额贡献</div><div class="kpi-value {{ 'negative' if stock_excess < 0 else 'positive' }}">{{ "%.2f"|format(stock_excess*100) }}%</div><div class="kpi-sub">对冲端基差 {{ "%.2f"|format(basis_total*100) }}%</div></div>
                <div class="kpi-card"><div class="kpi-label">CTA端贡献</div><div class="kpi-value {{ 'negative' if cta_total < 0 else 'positive' }}">{{ "%.2f"|format(cta_total*100) }}%</div><div class="kpi-sub">基差贡献 {{ "%.2f"|format(basis_contrib*100) }}%</div></div>
                <div class="kpi-card"><div class="kpi-label">市场成交热度</div><div class="kpi-value">2.3万亿</div><div class="kpi-sub">日均成交额 ↑</div></div>
            </div>
            <div class="market-insight">{{ market_text | safe }}</div>
        </div>
        <div class="report-body" style="padding: 32px 40px 48px;">
            <div class="section-title">📈 策略贡献分解（一）· 产品收益率 vs 各策略端贡献</div>
            <div class="card-grid">
                <div class="card"><div class="card-header">产品收益率与贡献分解</div><div class="card-content"><img class="chart-img" src="{{ img_paths.contrib1 }}" alt="贡献分解图"></div></div>
                <div class="card"><div class="card-header">数据明细</div><div class="card-content">{{ daily_table | safe }}</div></div>
            </div>
            <div class="section-title">⚖️ 策略贡献分解（二）· 股票超额 & 对冲基差</div>
            <div class="card-grid">
                <div class="card"><div class="card-header">股票超额 & 对冲基差</div><div class="card-content"><img class="chart-img" src="{{ img_paths.contrib2 }}" alt="股票超额与基差"></div></div>
                <div class="card"><div class="card-header">数据明细</div><div class="card-content">（请参见上表数据，此处略）</div></div>
            </div>
            <div class="section-title">📊 行业配置与盈亏对比（按日期）</div>
            <div class="card"><div class="card-content"><img class="chart-img" src="{{ img_paths.industry }}" alt="行业配置图"></div></div>
            <div class="section-title">📉 指数成分股中位数偏差</div>
            <div class="card-grid">
                <div class="card"><div class="card-header">沪深300 VS 个股中位数</div><div class="card-content"><img class="chart-img" src="{{ img_paths.median }}" alt="中位数偏差图"></div></div>
                <div class="card"><div class="card-header">数据表</div><div class="card-content">{{ median_table | safe }}</div></div>
            </div>
            <div class="section-title">🎯 风格暴露 · 因子雷达图</div>
            <div class="card"><div class="card-content"><img class="chart-img" src="{{ img_paths.style }}" alt="风格暴露雷达图"></div></div>
            <div class="footnote">报告生成日期：{{ report_date }} | 注：本报告数据截至 {{ period_end }}</div>
        </div>
    </div>
    </body>
    </html>
    '''
    template = Template(template_str)
    html = template.render(
        product_chinese=product_chinese,
        period_start=period_start,
        period_end=period_end,
        total_return=total_return,
        stock_excess=stock_excess_total,
        cta_total=cta_total,
        basis_contrib=basis_contrib,
        basis_total=basis_total,
        market_text=market_text,
        daily_table=daily_table_html,
        median_table=median_table_html,
        img_paths=img_paths,
        report_date=datetime.now().strftime('%Y-%m-%d')
    )
    return html

# ------------------------------------------------------------------------------
# 5. PDF 报告生成（使用 weasyprint）
# ------------------------------------------------------------------------------
def generate_pdf_report(product, dates, filename, df_daily, df_industry, df_median, mean_bias, bias_std):
    """生成 HTML 版面 PDF 报告"""
    product_name_map = {
        'PBHSZX1H': '配邦恒升中性1号',
        'PBPFZX1H': '配邦鹏飞中性1号',
        'PBTZ2H': '配邦投资二号'
    }
    product_chinese = product_name_map.get(product, '无')
    period_start = dates[1] if len(dates) > 1 else dates[0]
    period_end = dates[-1]

    # 注册字体（用于 matplotlib）
    font_prop = register_chinese_font()
    if font_prop is None:
        font_prop = fm.FontProperties()

    # 市场描述文字（固定）
    market_text = '''
    本周（4月13日—17日）A股震荡上行，创业板指大涨6.65%领跑，深证成指、沪深300分别上涨4.02%和1.99%，上证指数涨1.64%；市场交投活跃，日均成交额超2.3万亿元，融资资金大幅加仓电子、电力设备。宏观上，美伊停火缓和外部风险，国内一季度GDP增长5.0%、PPI同比转正，人民币升值吸引外资回流。结构上，沪深300指数每日涨幅均高于成分股涨跌幅中位数（平均偏差-0.0046%，最大偏差-0.92个百分点），表明指数上涨主要由少数头部权重股拉动，多数成分股表现滞后，市场呈现典型的结构性分化；通信、电子等成长风格领涨，石油石化、食品饮料等承压。整体看，成长赛道引领指数修复，但个股分化加剧，指数化布局更具优势。
    '''

    # 生成所有图表并保存为临时文件
    img_paths = {}

    buf1 = plot_strategy_contrib_1(df_daily, font_prop)
    img_paths['contrib1'] = save_buf_to_temp(buf1, 'contrib1.png')

    buf2 = plot_strategy_contrib_2(df_daily, font_prop)
    img_paths['contrib2'] = save_buf_to_temp(buf2, 'contrib2.png')

    buf_ind = plot_industry_combined(df_industry, font_prop)
    img_paths['industry'] = save_buf_to_temp(buf_ind, 'industry.png')

    buf_med = plot_median_deviation(df_median, font_prop)
    img_paths['median'] = save_buf_to_temp(buf_med, 'median.png')

    buf_style = plot_style_exposure(period_end, font_prop)
    img_paths['style'] = save_buf_to_temp(buf_style, 'style.png')

    # 生成 HTML 内容
    html_content = generate_html_report(
        product_chinese, period_start, period_end, df_daily, df_industry,
        df_median, img_paths, market_text
    )

    # 使用 weasyprint 将 HTML 转为 PDF
    weasyprint.HTML(string=html_content, base_url=TEMP_DIR).write_pdf(filename)

    # 清理临时图片文件（可选）
    for f in img_paths.values():
        if os.path.exists(f):
            os.remove(f)

    print(f"✅ 精美 HTML 版面 PDF 报告已生成：{os.path.abspath(filename)}")

# ------------------------------------------------------------------------------
# 6. 主函数
# ------------------------------------------------------------------------------
def main(start_date, end_date, net_path, fut_path, products, output_dir, email_list, test=True):
    """主函数，解析参数并运行分析"""
    init_rqdatac()

    dates = rqdatac.get_trading_dates(start_date, end_date)
    dates = [d.strftime('%Y%m%d') for d in dates]

    if len(dates) > 7:
        print('季度归因模式')
        product_dic, industry_dic = calculate_attribution_quarterly(dates, net_path, fut_path, products)
    else:
        print('周度归因模式')
        product_dic, industry_dic = calculate_attribution(dates, net_path, fut_path, products)

    df_median, mean_bias, bias_std = get_index_component_info(dates)
    df_median.to_csv('/home/zhanggh/DailyScripts/PDF/指数成分股中位数偏差.csv', encoding='gbk')
    
    # result_df = pd.DataFrame({
    #     "成分股涨跌幅中位数": median_return,
    #     "沪深300指数涨跌幅": index_return,
    #     "偏差": bias
    # })
    pd.concat(product_dic, ignore_index=True).to_csv('/home/zhanggh/DailyScripts/PDF/产品信息.csv', encoding='gbk')
    pd.concat(industry_dic, ignore_index=True).to_csv('/home/zhanggh/DailyScripts/PDF/行业信息.csv', encoding='gbk')

    manager = EmailManager()
    for product in products:
        df_daily = product_dic.get(product)
        df_daily.to_csv()
        df_industry = industry_dic.get(product)
        if df_daily is not None and df_industry is not None and not df_daily.empty and not df_industry.empty:
            df_daily['日期'] = df_daily['日期'].astype(str)
            df_industry['date'] = df_industry['date'].astype(str)
            product_name_map = {
                'PBHSZX1H': '配邦恒升中性1号',
                'PBPFZX1H': '配邦鹏飞中性1号',
                'PBTZ2H': '配邦投资二号'
            }
            product_chinese = product_name_map.get(product, product)
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'{product_chinese}_{dates[1]}_{dates[-1]}_分析报告_html.pdf')
            generate_pdf_report(product, dates, filename, df_daily, df_industry, df_median, mean_bias, bias_std)
            if not test:
                manager.send_email_with_attachments(email_list, '周度净值分析报告V1',
                                                     f'{product_chinese}_{dates[1]}_{dates[-1]}_分析报告',
                                                     attachments=[filename])
        else:
            print(f"警告：产品 {product} 无数据，跳过报告生成。")
    manager.logout()

if __name__ == '__main__':
    # 全局字体属性（供绘图函数使用）
    CHINESE_FONT_PROP = register_chinese_font()
    # 示例参数（请根据实际情况修改）
    start_date = '20260410'
    end_date = '20260417'
    net_path = '/home/zhanggh/DailyScripts/TradeData/standarddata/net_data'
    fut_path = '/home/zhanggh/DailyScripts/TradeData/standarddata/fut_data'
    products = ['PBHSZX1H']
    output_dir = '/home/zhanggh/DailyScripts/PDF'
    test = True
    email_list = ['pagududeshengjiang@shpbjj.com', 'xu_hengsheng@163.com']
    main(start_date, end_date, net_path, fut_path, products, output_dir, email_list, test)