#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
期货业绩归因分析报告生成脚本

功能：
- 读取产品净值、持仓、成交数据
- 计算量化期货收益、对冲端收益、CTA收益
- 进行策略贡献分解（产品收益率 vs 超额/基差/CTA）
- 进行行业配置分析（权重对比、偏离、盈亏）
- 生成商务风格的 PDF 报告
"""

import os
import sys
import warnings
import datetime as dt
from datetime import datetime
from functools import partial
from io import BytesIO
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg

import rqdatac
from ultis.email_manager import *
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. 数据获取函数
# ------------------------------------------------------------------------------

def init_rqdatac():
    """初始化米筐数据接口（请替换为您的账号信息）"""
    rqdatac.init(
        username="license",
        password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30="
    )


def get_fut_price_info(date, codes):
    """获取指定日期期货收盘价和结算价信息"""
    # 获取所有期货合约基本信息
    fut_mkt_df = rqdatac.all_instruments(type='Future', market='cn', date=date)
    # 筛选中金所股指期货和国债期货
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
        # 正常获取当日数据
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
        # 如果当日无数据，取前一交易日价格作为昨价，并从本地 CSV 中读取当日收盘价
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

        # 从本地 CSV 读取当日收盘价和成交量等信息
        data_path = f'/home/zhanggh/testscripts/test_data/{date}_data.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0).rename(columns={'order_book_id': 'code'})
            data = data.drop_duplicates(subset=None, keep='last', ignore_index=True)
            data['datetime'] = data['datetime'].astype(str)
            data[['date', 'time']] = data['datetime'].str.split(' ', expand=True)
            data['date'] = data['date'].str.replace('-', '')
            data['time'] = data['time'].str.replace(':', '')
            # 取15:00收盘价
            close_data = data[data['time'] == '150000'].reset_index()
            fut_mkt_df = pd.merge(fut_mkt_df, close_data[['code', 'close']], on='code', how='left')
            # 计算成交均价作为结算价
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

    # 处理成交记录
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

    # 处理持仓记录
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
        df['期货户出入金'] = df['期货户入金'] - df['期货户出金']   # 出入金净额
        # 获取指数价格
        try:
            idx_price = rqdatac.get_price('000300.XSHG', start_date=date, end_date=date).reset_index().loc[0, 'close']
        except:
            idx_price = rqdatac.current_minute(['000300.XSHG'], skip_suspended=False).reset_index().loc[0, 'close']
        # 获取期货价格
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
        # 如果当天没有数据，使用实时数据 + 前一日收盘
        stk = rqdatac.current_minute(stk_codes, fields=['close'], skip_suspended=False).reset_index()
        prev = rqdatac.get_price(stk_codes, start_date=preday, end_date=preday, frequency='1d',
                                  fields=['close'], adjust_type='pre', skip_suspended=False,
                                  market='cn', expect_df=True, time_slice=None).reset_index()
        prev = prev[['order_book_id', 'close']].rename(columns={'close': 'prev_close'})
        stk = pd.merge(stk[['order_book_id', 'close']], prev, on='order_book_id', how='left')
    return stk[['order_book_id', 'close', 'prev_close']]


def deal_industry_info(date, product, benchmark='000300.XSHG'):
    """获取指数行业分布和持仓行业分布"""
    # 获取基准指数成分股权重
    pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
    bench_weights = rqdatac.index_weights_ex(benchmark, start_date=pre_date, end_date=pre_date, market='cn').loc[pre_date].reset_index()
    bench_codes = bench_weights['order_book_id'].unique().tolist()
    bench_industry = rqdatac.get_instrument_industry(order_book_ids=bench_codes, date=pre_date)['first_industry_name'].reset_index()
    bench_industry = bench_industry.rename(columns={'first_industry_name': 'industry'})
    # 合并权重和行业
    bench_df = pd.merge(bench_weights, bench_industry, on='order_book_id')
    # 添加涨跌幅
    stk_df = stk_mkt_close(date)
    stk_df['pnl'] = stk_df['close'] / stk_df['prev_close'] - 1
    bench_df = pd.merge(bench_df, stk_df[['order_book_id', 'pnl']], on='order_book_id', how='left')
    bench_df['benchmark_pnl'] = bench_df['weight'] * bench_df['pnl']
    # 按行业汇总
    bench_ind = bench_df.groupby('industry').agg(
        bench_w=('weight', 'sum'),
        benchmark_pnl=('benchmark_pnl', 'sum')
    ).reset_index()
    bench_ind['benchmark_pnl'] = bench_ind['benchmark_pnl'].apply(lambda x: f'{x:.2%}')

    # 读取实际持仓
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
        # 获取行业分类（中证800）
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
        # 合并基准和实际
        res = pd.merge(bench_ind, pos_ind, on='industry', how='outer')
        res['pos_w'] = res['pos_w'].fillna(0)
        res['pos_pnl'] = res['pos_pnl'].fillna('0.00%')
        res['bench_w'] = res['bench_w'].fillna(0)
        res['benchmark_pnl'] = res['benchmark_pnl'].fillna('0.00%')
        res['date'] = date
        return res
    else:
        # 如果没有持仓文件，返回空 DataFrame
        return pd.DataFrame()


def get_period_dates(dates, freq='week'):
    """
    将日期列表按指定频率分组，返回每个周期的结束日期（用于季度归因）
    此处简化，默认将日期列表按自然周/月分组，但实际使用中需自定义
    在代码中使用了外部 CSV 文件 date.csv，这里我们改为动态计算
    """
    # 为了保持与原代码一致，这里简单返回原始日期列表（只支持周度）
    # 实际生产环境中应根据需要实现分组逻辑
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

        # 保存日度数据
        daily_cols = ['日期', '产品', '产品收益率', '股票端对产品的贡献', '超额对产品的贡献',
                      '对冲端对产品的贡献', '基差对产品的贡献', 'CTA端对产品的贡献',
                      '股票端超额', '对冲端基差收益']
        tmp_daily = tmp[daily_cols].iloc[1:].reset_index(drop=True)
        product_dic[product] = tmp_daily

        # 行业配置数据
        act_dates = dates[1:]   # 从第二天开始
        res = pd.DataFrame()
        for date in act_dates:
            ind_df = deal_industry_info(date, product)
            if not ind_df.empty:
                res = pd.concat([res, ind_df])
        industry_dic[product] = res

    return product_dic, industry_dic


def calculate_attribution_quarterly(dates, net_path, fut_path, products):
    """计算业绩归因（季度汇总）"""
    # 获取季度末日期（实际项目中需自定义，这里简化处理，直接用原日期）
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
        print(tmp)

        # 按季度汇总
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

        # 行业配置
        act_dates = sec_dates[1:]
        res = pd.DataFrame()
        for date in act_dates:
            ind_df = deal_industry_info(date, product)
            if not ind_df.empty:
                res = pd.concat([res, ind_df])
        industry_dic[product] = res

    return product_dic, industry_dic


def get_index_component_info(dates, INDEX_CODE = "000300.XSHG"):
    """获取指数成分股中位数偏差（原函数）"""
    START_DATE = dates[0]
    END_DATE = dates[-1]
    component_stocks = rqdatac.index_components(INDEX_CODE, date=START_DATE)
    all_instruments = component_stocks + [INDEX_CODE]
    price_data = rqdatac.get_price(all_instruments, start_date=START_DATE, end_date=END_DATE, fields=["close"], adjust_type="pre", frequency="1d")
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

    print(result_df.head())
    mean_bias = bias.mean()
    bias_std = bias.std()
    print(f'平均偏差:{mean_bias}，偏差标准差:{bias_std}')
    return result_df, mean_bias, bias_std


def register_chinese_font():
    """注册中文字体（跨平台）"""
    system = platform.system()
    font_path = None

    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break
    elif system == "Darwin":   # macOS
        candidates = [
            os.path.expanduser("~/Library/Fonts/Microsoft YaHei.ttf"),
            "/System/Library/Fonts/PingFang.ttc",
        ]
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
            # 注册到 reportlab
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            addMapping('SimHei', 0, 0, 'SimHei')
            addMapping('SimHei', 1, 0, 'SimHei')
            addMapping('SimHei', 0, 1, 'SimHei')
            addMapping('SimHei', 1, 1, 'SimHei')
            # matplotlib
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已加载中文字体: {font_path}")
            return 'SimHei', fm.FontProperties(fname=font_path)
        except Exception as e:
            print(f"❌ 字体加载失败: {e}")
    else:
        print("⚠️ 中文字体未找到，PDF 标题将使用英文字体")
    return 'Helvetica', None


def format_df_as_percent(df):
    """将 DataFrame 中的数值列格式化为百分比字符串"""
    df_formatted = df.copy()
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df_formatted[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    return df_formatted


def dataframe_to_image(df, title=""):
    """将 DataFrame 转换为图片（用于 PDF 嵌入）"""
    df_display = format_df_as_percent(df)
    fig, ax = plt.subplots(figsize=(10, 2 + len(df) * 0.6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_display.values, colLabels=df_display.columns,
                     cellLoc='center', loc='center',
                     colColours=['#e6f0ff'] * len(df_display.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.2, 1.6)
    for (i, j), cell in table.get_celld().items():
        if i == 0:   # header
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#1a56db')
        else:
            cell.set_text_props(color='black')
            cell.set_facecolor('white' if i % 2 == 1 else '#f9f9f9')
        if CHINESE_FONT_PROP:
            cell.set_text_props(fontproperties=CHINESE_FONT_PROP)
    if title:
        ax.set_title(title, fontproperties=CHINESE_FONT_PROP, fontsize=13, pad=4)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def plot_strategy_contrib_0(df, mean_bias, bias_std):
    """指数中位数偏差"""

    df_plot = df.copy()
    df_plot = df_plot.reset_index()
    df_plot['date'] = df_plot['date'].apply(lambda x: str(x).replace('-', '')[:8])
    dates = df_plot['date']
    df_plot = df_plot[['date', '成分股涨跌幅中位数', '沪深300指数涨跌幅', '偏差']]
    table_df = df_plot[['date', '成分股涨跌幅中位数', '沪深300指数涨跌幅', '偏差']].copy()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dates, df_plot['成分股涨跌幅中位数'], color='#1a56db', marker='o',
             label='成分股涨跌幅中位数', linewidth=2.5, markersize=8)
    ax1.plot(dates, df_plot['沪深300指数涨跌幅'], color='#2ecc71', marker='o',
             label='沪深300指数涨跌幅', linewidth=2.5, markersize=8)
    ax1.plot(dates, df_plot['偏差'], color='#e74c3c', marker='o',
             label='偏差', linewidth=2.5, markersize=8)
    ax1.set_ylabel('指数成分股中位数偏差', fontproperties=CHINESE_FONT_PROP, fontsize=11)
    ax1.set_title('指数成分股中位数偏差',fontproperties=CHINESE_FONT_PROP, fontsize=13)
    ax1.legend(loc='upper left', prop=CHINESE_FONT_PROP, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4, color='#e0e0e0')
    ax1.set_xticklabels(dates, rotation=45, fontsize=9, fontproperties=CHINESE_FONT_PROP)
    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout(pad=2.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf, dataframe_to_image(table_df, "数据表：指数成分股中位数偏差")


def plot_strategy_contrib_1(df):
    """策略贡献分解图（产品收益率 + 超额/基差/CTA）"""
    df_plot = df.copy()
    dates = df_plot['日期']
    table_df = df[['日期', '产品收益率', '超额对产品的贡献', '基差对产品的贡献', 'CTA端对产品的贡献']].copy()
    table_df.loc['合计'] = table_df.select_dtypes(include='number').sum()
    table_df.iloc[-1, 0] = '合计'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dates, df_plot['产品收益率'], color='#1a56db', marker='o',
             label='产品收益率', linewidth=2.5, markersize=8)
    width = 0.22
    x = range(len(dates))
    ax1.bar([i - width for i in x], df_plot['超额对产品的贡献'], width,
            color='#2ecc71', label='超额贡献', alpha=0.85)
    ax1.bar(x, df_plot['基差对产品的贡献'], width,
            color='#e74c3c', label='基差贡献', alpha=0.85)
    ax1.bar([i + width for i in x], df_plot['CTA端对产品的贡献'], width,
            color='#3498db', label='CTA贡献', alpha=0.85)
    ax1.set_ylabel('收益贡献', fontproperties=CHINESE_FONT_PROP, fontsize=11)
    ax1.set_title('策略贡献分解（一）：产品收益率 vs 各策略端贡献',
                  fontproperties=CHINESE_FONT_PROP, fontsize=13)
    ax1.legend(loc='upper left', prop=CHINESE_FONT_PROP, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4, color='#e0e0e0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45, fontsize=9, fontproperties=CHINESE_FONT_PROP)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout(pad=2.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf, dataframe_to_image(table_df, "数据表：策略贡献（一）")


def plot_strategy_contrib_2(df):
    """策略贡献分解图（股票超额 + 对冲基差）"""
    df_plot = df.copy()
    dates = df_plot['日期']
    df_plot['合计'] = df_plot['股票端超额'] + df_plot['对冲端基差收益']
    table_df = df[['日期', '股票端超额', '对冲端基差收益']].copy()
    table_df.loc['合计'] = table_df.select_dtypes(include='number').sum()
    table_df.iloc[-1, 0] = '合计'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(dates, df_plot['合计'], color='#1a56db', marker='o',
             label='标准中性策略', linewidth=2.5, markersize=8)
    width = 0.3
    x = range(len(dates))
    ax1.bar([i - width/2 for i in x], df_plot['股票端超额'], width,
            color='#2ecc71', label='股票端超额', alpha=0.85)
    ax1.bar([i + width/2 for i in x], df_plot['对冲端基差收益'], width,
            color='#e74c3c', label='对冲端基差收益', alpha=0.85)
    ax1.set_ylabel('收益贡献', fontproperties=CHINESE_FONT_PROP, fontsize=11)
    ax1.set_title('策略贡献分解（二）：股票超额 & 对冲基差',
                  fontproperties=CHINESE_FONT_PROP, fontsize=13)
    ax1.legend(loc='upper left', prop=CHINESE_FONT_PROP, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4, color='#e0e0e0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, rotation=45, fontsize=9, fontproperties=CHINESE_FONT_PROP)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.2f}%'))
    plt.tight_layout(pad=2.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf, dataframe_to_image(table_df, "数据表：策略贡献（二）")


def plot_industry_combined_by_date(df):
    """行业配置：权重对比、偏离、盈亏对比（按日期分页）"""
    dates = sorted(df['date'].unique())
    if len(dates) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, '无行业数据', transform=ax.transAxes, ha='center', va='center', fontsize=11)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf

    n_dates = len(dates)
    fig, axes = plt.subplots(n_dates, 3, figsize=(24, 8 * n_dates), sharey='row')
    if n_dates == 1:
        axes = [axes]

    COLOR_BENCH = '#e74c3c'    # 基准权重（红）
    COLOR_POS = '#2ecc71'      # 实际仓位（绿）
    COLOR_DEV = '#3498db'      # 权重偏离（蓝）
    COLOR_BENCH_PNL = '#3498db'  # 基准盈亏（蓝）
    COLOR_POS_PNL = '#e74c3c'    # 实际盈亏（红）

    for idx, date in enumerate(dates):
        sub = df[df['date'] == date].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('pos_w', ascending=True)

        industries = sub['industry'].values
        bench_w = sub['bench_w'].values
        pos_w = sub['pos_w'].values
        weight_deviation = pos_w - bench_w
        bench_pnl = [float(p.replace('%', '')) / 100 for p in sub['benchmark_pnl']]
        pos_pnl = [float(p.replace('%', '')) / 100 for p in sub['pos_pnl']]
        y_pos = range(len(industries))
        bar_height = 0.35

        # 第一列：权重对比
        ax_left = axes[idx][0] if n_dates > 1 else axes[0]
        ax_left.barh(y_pos, bench_w, height=bar_height, color=COLOR_BENCH, edgecolor='black', label='基准权重')
        ax_left.barh([y + bar_height for y in y_pos], pos_w, height=bar_height, color=COLOR_POS, edgecolor='black', label='实际仓位')
        ax_left.set_yticks([y + bar_height/2 for y in y_pos])
        ax_left.set_yticklabels(industries, fontproperties=CHINESE_FONT_PROP, fontsize=9)
        ax_left.set_xlabel('权重', fontproperties=CHINESE_FONT_PROP, fontsize=10)
        ax_left.set_title(f'行业权重（{date}）', fontproperties=CHINESE_FONT_PROP, fontsize=11)
        ax_left.legend(loc='lower right', prop=CHINESE_FONT_PROP, fontsize=9)
        ax_left.grid(axis='x', linestyle='--', alpha=0.4, color='#e0e0e0')
        ax_left.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))

        # 第二列：权重偏离
        ax_mid = axes[idx][1] if n_dates > 1 else axes[1]
        colors_dev = [COLOR_POS if d >= 0 else COLOR_BENCH for d in weight_deviation]
        ax_mid.barh(y_pos, weight_deviation, height=bar_height, color=colors_dev, edgecolor='black')
        ax_mid.set_xlabel('权重偏离（实际 - 基准）', fontproperties=CHINESE_FONT_PROP, fontsize=10)
        ax_mid.set_title(f'权重偏离（{date}）', fontproperties=CHINESE_FONT_PROP, fontsize=11)
        ax_mid.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax_mid.grid(axis='x', linestyle='--', alpha=0.4, color='#e0e0e0')
        ax_mid.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))
        max_dev = max(abs(d) for d in weight_deviation) if weight_deviation.size > 0 else 0.001
        ax_mid.set_xlim(-max_dev * 1.2, max_dev * 1.2)

        # 第三列：盈亏对比
        ax_right = axes[idx][2] if n_dates > 1 else axes[2]
        ax_right.barh(y_pos, bench_pnl, height=bar_height, color=COLOR_BENCH_PNL, edgecolor='black', alpha=0.8, label='基准盈亏')
        ax_right.barh([y + bar_height for y in y_pos], pos_pnl, height=bar_height, color=COLOR_POS_PNL, edgecolor='black', alpha=0.8, label='实际盈亏')
        ax_right.set_xlabel('盈亏', fontproperties=CHINESE_FONT_PROP, fontsize=10)
        ax_right.set_title(f'行业盈亏（{date}）', fontproperties=CHINESE_FONT_PROP, fontsize=11)
        ax_right.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax_right.grid(axis='x', linestyle='--', alpha=0.4, color='#e0e0e0')
        ax_right.legend(loc='lower right', prop=CHINESE_FONT_PROP, fontsize=9)
        ax_right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))
        all_vals = bench_pnl + pos_pnl
        max_val = max(abs(v) for v in all_vals) if all_vals else 0.001
        ax_right.set_xlim(-max_val * 1.3, max_val * 1.3)

    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

# 临时加图
def get_pic():
    img_style = mpimg.imread("/home/zhanggh/remain/position_backtest/custom_portfolio_202604211632_style_exposure.png")
    img_industry = mpimg.imread("/home/zhanggh/remain/position_backtest/custom_portfolio_202604211632_industry_weight.png")
    return img_style, img_industry



# ------------------------------------------------------------------------------
# 4. PDF 报告生成
# ------------------------------------------------------------------------------

def generate_pdf_report(product, dates, filename, df_daily, df_industry, df, mean_bias, bias_std):
    """生成商务风格的 PDF 报告"""
    product_name_map = {
        'PBHSZX1H': '配邦恒升中性1号',
        'PBPFZX1H': '配邦鹏飞中性1号',
        'PBTZ2H': '配邦投资二号'
    }
    product_chinese = product_name_map.get(product, '无')

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            leftMargin=50, rightMargin=50,
                            topMargin=40, bottomMargin=40)

    # 定义样式
    title_style = ParagraphStyle(
        'Title', fontName=CHINESE_FONT, fontSize=20, alignment=1,
        spaceAfter=18, leading=26, textColor='#2c3e50'
    )
    heading2_style = ParagraphStyle(
        'H2', fontName=CHINESE_FONT, fontSize=14, spaceBefore=14,
        spaceAfter=10, leading=18, textColor='#1a56db'
    )
    normal_style = ParagraphStyle(
        'Normal', fontName=CHINESE_FONT, fontSize=11, leading=14, spaceAfter=6
    )
    context_style = ParagraphStyle(
        'Normal', fontName=CHINESE_FONT, fontSize=10, leading=14, spaceAfter=6
    )

    story = []
    story.append(Paragraph(f"{product_chinese} {dates[1]}-{dates[-1]} 产品分析报告", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("本报告基于产品历史数据进行策略贡献与行业配置分析，仅供参考。", normal_style))
    story.append(Spacer(1, 14))
    context = '本周（4月13日—17日）A股震荡上行，创业板指大涨6.65%领跑，深证成指、沪深300分别上涨4.02%和1.99%，上证指数涨1.64%；市场交投活跃，日均成交额超2.3万亿元，融资资金大幅加仓电子、电力设备。宏观上，美伊停火缓和外部风险，国内一季度GDP增长5.0%、PPI同比转正，人民币升值吸引外资回流。结构上，沪深300指数每日涨幅均高于成分股涨跌幅中位数（平均偏差-0.0046%，最大偏差-0.92个百分点），表明指数上涨主要由少数头部权重股拉动，多数成分股表现滞后，市场呈现典型的结构性分化；通信、电子等成长风格领涨，石油石化、食品饮料等承压。整体看，成长赛道引领指数修复，但个股分化加剧，指数化布局更具优势。'
    story.append(Paragraph(context, context_style))
    story.append(Spacer(1, 14))


    # 图1
    img1, tbl1 = plot_strategy_contrib_1(df_daily)
    story.append(Paragraph("1. 策略贡献分解（一）：产品收益率 vs 各策略端贡献", heading2_style))
    story.append(Image(img1, width=6.8*inch, height=3*inch))
    story.append(Spacer(1, 6))
    story.append(Image(tbl1, width=6.8*inch, height=5*inch))
    story.append(Spacer(1, 18))
    story.append(PageBreak())

    # 图2
    img2, tbl2 = plot_strategy_contrib_2(df_daily)
    story.append(Paragraph("2. 策略贡献分解（二）：股票超额 & 对冲基差", heading2_style))
    story.append(Image(img2, width=6.8*inch, height=3*inch))
    story.append(Spacer(1, 6))
    story.append(Image(tbl2, width=6.8*inch, height=5*inch))
    story.append(Spacer(1, 18))
    story.append(PageBreak())

    # 图3
    story.append(Paragraph("3. 行业配置与盈亏对比（按日期）", heading2_style))
    img3 = plot_industry_combined_by_date(df_industry)
    story.append(Image(img3, width=6.8*inch, height=9*inch))

    # 图4
    img_4, tbl4 = plot_strategy_contrib_0(df, mean_bias, bias_std) 
    story.append(Paragraph("4. 指数中位数偏差", heading2_style))
    story.append(Image(img_4, width=6.8*inch, height=3*inch))
    story.append(Spacer(1, 6))
    story.append(Image(tbl4, width=6.8*inch, height=5*inch))
    story.append(Spacer(1, 18))
    story.append(PageBreak())

    # 图1,2
    img_style_paths = f"/home/zhanggh/remain/position_backtest/custom_portfolio_{dates[-1]}_style_exposure.png"
    img_style = mpimg.imread(img_style_paths)

    # img_industry = mpimg.imread("/home/zhanggh/remain/position_backtest/custom_portfolio_202604211632_industry_weight.png")
    story.append(Paragraph("5. 风格暴露", heading2_style))
    story.append(Image(img_style_paths, width=6.8*inch, height=3*inch))
    story.append(Spacer(1, 6))

    # story.append(Paragraph("5. 风格暴露", heading2_style))
    # story.append(Image(tbl1, width=6.8*inch, height=5*inch))
    # story.append(Spacer(1, 18))
    # story.append(PageBreak())

    # 报告尾部
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"报告生成日期: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    story.append(Paragraph(f"注：本报告数据截至 {dates[-1]}", normal_style))

    doc.build(story)
    print(f"✅ 商务版 PDF 报告已生成：{os.path.abspath(filename)}")


# ------------------------------------------------------------------------------
# 5. 主函数
# ------------------------------------------------------------------------------

def main(start_date, end_date, net_path, fut_path, products, output_dir, email_list, test=True):
    """主函数，解析参数并运行分析"""
    # import argparse
    # parser = argparse.ArgumentParser(description='期货业绩归因分析报告生成')
    # parser.add_argument('--start_date', default='20251231', help='起始日期 (YYYYMMDD)')
    # parser.add_argument('--end_date', default='20260331', help='结束日期 (YYYYMMDD)')
    # parser.add_argument('--net_path', default='/home/zhanggh/DailyScripts/TradeData/standarddata/net_data',
    #                     help='净值数据路径')
    # parser.add_argument('--fut_path', default='/home/zhanggh/DailyScripts/TradeData/standarddata/fut_data',
    #                     help='期货数据路径')
    # parser.add_argument('--products', nargs='+', default=['PBHSZX1H'],
    #                     help='产品列表，默认 PBHSZX1H')
    # parser.add_argument('--output_dir', default='/home/zhanggh/DailyScripts/PDF',
    #                     help='输出目录')
    # args = parser.parse_args()
    # start = args.start_date
    # end = args.end_date

    # 初始化米筐
    init_rqdatac()

    # 获取交易日列表
    dates = rqdatac.get_trading_dates(start_date, end_date)
    dates = [d.strftime('%Y%m%d') for d in dates]

    # 判断周期（如果超过7天则按季度处理，否则按周）
    if len(dates) > 7:
        print('季度归因模式')
        product_dic, industry_dic = calculate_attribution_quarterly(dates, net_path, fut_path, products)
    else:
        print('周度归因模式')
        product_dic, industry_dic = calculate_attribution(dates, net_path, fut_path, products)
    df, mean_bias, bias_std = get_index_component_info(dates)
    print(df)
    print(mean_bias)
    print(bias_std)

    # 生成 PDF
    manager = EmailManager()
    for product in products:
        df_daily = product_dic.get(product)
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
            filename = os.path.join(output_dir, f'{product_chinese}_{dates[1]}_{dates[-1]}_分析报告.pdf')
            generate_pdf_report(product, dates, filename, df_daily, df_industry, df, mean_bias, bias_std)
            if not test:
                manager.send_email_with_attachments(email_list, '周度净值分析报告V1', f'{product_chinese}_{dates[1]}_{dates[-1]}_分析报告', attachments=[filename])
        else:
            print(f"警告：产品 {product} 无数据，跳过报告生成。")
        manager.logout()
        


if __name__ == '__main__':
    # 全局字体属性（供绘图函数使用）
    CHINESE_FONT, CHINESE_FONT_PROP = register_chinese_font()
    '''
    
    parser = argparse.ArgumentParser(description='期货业绩归因分析报告生成')
    parser.add_argument('--start_date', default='20251231', help='起始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', default='20260331', help='结束日期 (YYYYMMDD)')
    parser.add_argument('--net_path', default='/home/zhanggh/DailyScripts/TradeData/standarddata/net_data',
                        help='净值数据路径')
    parser.add_argument('--fut_path', default='/home/zhanggh/DailyScripts/TradeData/standarddata/fut_data',
                        help='期货数据路径')
    parser.add_argument('--products', nargs='+', default=['PBHSZX1H'],
                        help='产品列表，默认 PBHSZX1H')
    parser.add_argument('--output_dir', default='/home/zhanggh/DailyScripts/PDF',
                        help='输出目录')
    args = parser.parse_args()
    '''
    start_date = '20260410'
    end_date = '20260417'
    net_path = '/home/zhanggh/DailyScripts/TradeData/standarddata/net_data'
    fut_path = '/home/zhanggh/DailyScripts/TradeData/standarddata/fut_data'
    products = ['PBHSZX1H']
    output_dir = '/home/zhanggh/DailyScripts/PDF'
    test = True
    email_list = ['pagududeshengjiang@shpbjj.com', 'xu_hengsheng@163.com'] # 
    main(start_date, end_date, net_path, fut_path, products, output_dir, email_list, test)