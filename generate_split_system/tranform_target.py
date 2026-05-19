"""
tranform_target.py
读取模型预测数据 → 组合优化 → 生成各 target 的权重文件

所有可调参数均从 target_config.yaml 读取，无需修改代码。
"""
import io
import os
import sys
import contextlib
import logging
import warnings
import datetime as dt

import yaml
import rqdatac
import pandas as pd
import numpy as np
from pathlib import Path

# 自动把 D:\code 加入 Python 搜索路径
sys.path.append(str(Path(__file__).parent.parent))

# add_factor 延迟导入：rqfactors 在 import 时就会调 rqdatac API，
# 必须等 rqdatac.init() 之后才能安全导入（见 _get_model_data_and_deal 方法内）
from ultis.email_manager import EmailManager
from ultis.stock_data_client import StockDataClient
import ultis.portfolio_optimizer_re_test as portfolio_optimizer
import ultis.factor_explosure_test as factor_explosure

warnings.filterwarnings('ignore')

# 项目内模块
from config_loader import Config
from logger import setup_logger, get_logger


class TransformTarget:
    """
    目标权重生成器

    流程:
        1. 下载黑名单邮件
        2. 计算因子暴露（一次性，所有 target 共用）
        3. 对每个 unique_id: 模型筛选 → 组合优化 → 输出 target CSV
    """

    def __init__(self, config: Config, date: str):
        # ---- 从配置读取所有参数 ----
        self.model_path = config.get('model_path')
        self.target_path = config.get('target_path')
        self.database_path = config.get('DATABASE_PATH')
        self.blacklist_path = config.get('blacklist_path')
        self.model_file_pattern = config.get('model_file_pattern',
                                              'df_test_PB_V0422_{pre_date}.parquet')
        self.parameters = config.get('parameters', {})
        self.factor_explosure_param = config.get('factor_explosure_param', {})

        # 运行时参数（全部可调，有默认值兜底）
        rt = config.get('runtime', {})
        self._rt_placeholder_date = rt.get('blacklist_placeholder_date', '20250714')
        self._rt_download_limit = rt.get('blacklist_download_limit', 20)
        self._rt_alert_recipients = rt.get('blacklist_alert_recipients',
                                            ['15556235305@163.com',
                                             'pagududeshengjiang@shpbjj.com'])
        sf = rt.get('stock_filter', {})
        self._rt_min_turnover = sf.get('min_turnover_ma10', 25000000)
        self._rt_min_market_cap = sf.get('min_market_cap_ma3', 2500000000)
        self._rt_score_col = rt.get('model_score_column', 'Scorpio_ezavg_grouped_scaled')
        self._rt_score_rename = rt.get('model_score_rename', 'score')
        lr = rt.get('limit_ratios', {})
        self._rt_innovation_up = lr.get('innovation_up', 1.1985)
        self._rt_innovation_down = lr.get('innovation_down', 0.801)
        self._rt_normal_up = lr.get('normal_up', 1.0985)
        self._rt_normal_down = lr.get('normal_down', 0.901)

        # 日期
        self.date = date
        self._date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
        self.pre_date = rqdatac.get_previous_trading_date(date).strftime('%Y%m%d')
        self._pre_date = f'{self.pre_date[:4]}-{self.pre_date[4:6]}-{self.pre_date[6:8]}'

        # 日志
        self._logger = get_logger('tranform_target')
        self._logger.info(f'初始化完成 | 日期={self.date}  前交易日={self.pre_date}')

        # 数据库客户端（从配置读取路径）
        self._client = StockDataClient(data_path=self.database_path)

        # 下载黑名单
        self._download_blackfile_from_email()

        # 因子暴露（一次性计算，抑制外部模块冗长输出）
        self.stock_factors, self.index_exposures = self._get_factor_explosure()

    # ====================================================================
    # 黑名单邮件
    # ====================================================================
    def _download_blackfile_from_email(self):
        manager = EmailManager()

        cms_file = f'{self.blacklist_path}/cms_long_short_black_list_{self.date}.xlsx'
        if not os.path.exists(cms_file):
            self._logger.info('下载招商 DMA 黑名单...')
            manager.download_attachments_by_keyword(
                ['cms_long_short_black_list', self.date],
                save_dir=self.blacklist_path,
                file_extensions=['.xlsx'],
                limit=self._rt_download_limit,
            )

        restriction_file = f'{self.blacklist_path}/Restriction_List_{self.date}.xlsx'
        if not os.path.exists(restriction_file):
            self._logger.info('下载中信里昂限制名单...')
            manager.download_attachments_by_keyword(
                ['Restriction List', self.date],
                save_dir=self.blacklist_path,
                file_extensions=['.xlsx'],
                limit=self._rt_download_limit,
            )
            if not os.path.exists(restriction_file):
                self._logger.warning('中信里昂黑名单缺失，发送提醒邮件')
                manager.send_email_with_attachments(
                    to=self._rt_alert_recipients,
                    subject='中信里昂黑名单缺失',
                    body='未收到中信里昂黑名单邮件',
                )

        manager.logout()

    # ====================================================================
    # 因子暴露
    # ====================================================================
    def _get_factor_explosure(self):
        try:
            stock_factors = pd.read_parquet(rf'{self.database_path}\factors\factors_post_1d\adjusted_1800_vs_300exposures.parquet').set_index(['date', 'order_book_id'])
            index_exposures = pd.read_parquet(rf'{self.database_path}\factors\factors_post_1d\adjusted_300_index_exposures.parquet')
            self._logger.info(
                f'从服务器获取因子暴露完成 | 个股={stock_factors.shape}  指数暴露={index_exposures.shape}')
        except Exception as e:
            self._logger.error(e)
            sample_range = self.factor_explosure_param.get('sample_range')
            bench_mark = self.factor_explosure_param.get('bench_mark')
            start_date = self.factor_explosure_param.get('factor_start_date')
            window = self.factor_explosure_param.get('factor_rolling_window')

            self._logger.info(
                f'主动计算因子暴露 | sample={sample_range}  bench={bench_mark}  '
                f'start={start_date}  window={window}d')

            # 抑制外部模块的冗长 print 输出
            with contextlib.redirect_stdout(io.StringIO()) as _:
                stock_factors, index_exposures = factor_explosure.main(
                    sample_range, bench_mark, start_date,
                    self.database_path, window)
        return stock_factors, index_exposures

    # ====================================================================
    # 个股基准权重 & 动态单票上限
    # ====================================================================
    def _get_index_weight_limits(self, sample_range, bench_mark,
                                  stock_weight_multiplier, stock_weight_fallback,
                                  stock_weight_cap=None):
        try:
            return factor_explosure.compute_index_weight_limits(
                sample_range=sample_range,
                bench_mark=bench_mark,
                start=self._pre_date,
                end=self._pre_date,
                data_path=self.database_path,
                stock_weight_multiplier=float(stock_weight_multiplier),
                stock_weight_fallback=float(stock_weight_fallback),
                stock_weight_cap=(None if stock_weight_cap is None
                                  else float(stock_weight_cap)),
            )
        except Exception as exc:
            self._logger.warning(f'个股基准权重计算失败，回退固定上限 | {exc!r}')
            return None

    # ====================================================================
    # Benchmark 行业 & 权重（本地数据库口径）
    # ====================================================================
    def _get_benchmark_industry_weights(self, bench_mark):
        client = StockDataClient(data_path=self.database_path)

        bw_wide = client.get_stock_index_comments_weights(
            bench_mark, start=self._pre_date, end=self._pre_date)
        bw_wide = bw_wide.copy()
        bw_wide.index = pd.to_datetime(bw_wide.index)
        bw = (bw_wide.reset_index()
              .rename(columns={'index': 'date'})
              .melt(id_vars='date', var_name='order_book_id', value_name='weight'))
        bw['date'] = pd.to_datetime(bw['date'])
        bw['weight'] = pd.to_numeric(bw['weight'], errors='coerce')
        bw = bw.dropna(subset=['date', 'order_book_id', 'weight'])
        bw = bw[bw['weight'] > 0].reset_index(drop=True)
        codes = bw['order_book_id'].unique().tolist()

        industry_path = os.path.join(
            self.database_path,
            'stocks/basics/all_instruments_level1_industry.feather')
        ind = pd.read_feather(industry_path)
        ind['date'] = pd.to_datetime(ind['date'])
        ind = ind[(ind['date'] == pd.to_datetime(self._pre_date))
                   & (ind['order_book_id'].isin(codes))
                   ][['order_book_id', 'industry']].drop_duplicates(
                       subset=['order_book_id'], keep='last')

        result = pd.merge(bw, ind, on='order_book_id', how='outer')
        result['date'] = pd.to_datetime(self._pre_date)
        return result[['date', 'order_book_id', 'weight', 'industry']]

    # ====================================================================
    # 黑名单
    # ====================================================================
    def _get_blacklist(self, black_list, forbid_lst):
        blacklist = []
        if black_list:
            resolved = black_list.replace(self._rt_placeholder_date, f'{self.date}')
            data = pd.read_excel(f'{self.blacklist_path}/{resolved}')
            data['pair_id'] = data['pair_id'].apply(lambda x: str(x).split('@')[0])
            blacklist = data['pair_id'].unique().tolist()
        combined = list(set(blacklist + forbid_lst))

        preview = combined[:5]
        detail = ', '.join(preview) if preview else '(无)'
        if len(combined) > 5:
            detail += f' ...共{len(combined)}只'
        else:
            detail += f'  共{len(combined)}只'
        self._logger.info(f'  黑名单: {detail}')
        return combined

    # ====================================================================
    # 获涨停股票
    # ====================================================================
    def _get_near_limit_codes(self, codes):
        price_data = rqdatac.get_price(
            codes, start_date=self._pre_date, end_date=self._pre_date,
            frequency='1d', fields=None, adjust_type='pre', skip_suspended=False,
            market='cn', expect_df=True, time_slice=None).reset_index()

        def calc_limits(row):
            prefix = row['order_book_id'][:2]
            if prefix in ['68', '30']:
                return pd.Series([
                    row['close'] * self._rt_innovation_up,
                    row['close'] * self._rt_innovation_down])
            return pd.Series([
                row['close'] * self._rt_normal_up,
                row['close'] * self._rt_normal_down])

        price_data[['next_lmt_up', 'next_lmt_down']] = price_data.apply(
            calc_limits, axis=1)
        close_price = rqdatac.current_minute(codes).reset_index()
        merged = pd.merge(
            price_data[['order_book_id', 'next_lmt_up']],
            close_price[['order_book_id', 'close']],
            on='order_book_id', how='left')
        near = merged[merged['close'] >= merged['next_lmt_up']].reset_index(drop=True)
        return near['order_book_id'].unique().tolist()

    # ====================================================================
    # 模型数据筛选
    # ====================================================================
    def _get_model_data_and_deal(self, black_list, forbid_lst, sample_range=None):
        blacklist = self._get_blacklist(black_list, forbid_lst)

        model_file = self.model_file_pattern.replace('{pre_date}', self.pre_date)
        model_path_full = os.path.join(self.model_path, model_file)
        if not os.path.exists(model_path_full):
            raise FileNotFoundError(f'模型文件不存在: {model_path_full}')

        self._logger.info(f'  加载模型: {model_file}')
        model_data = pd.read_parquet(model_path_full)
        model_data = model_data.reset_index(drop=True).rename(
            columns={self._rt_score_col: self._rt_score_rename})

        # 过滤黑名单
        model_data = model_data[
            ~model_data['order_book_id'].isin(blacklist)].reset_index(drop=True)

        # 成交量和市值因子（延迟导入 add_factor，避免模块级 rqdatac API 调用）
        from ultis.dataproxy.rqfactors import add_factor
        sorted_data = add_factor(model_data, 'total_turnover_ma10',
                                  date_field='date', asset_field='order_book_id')
        sorted_data['prev_date'] = sorted_data['date'].apply(
            rqdatac.get_previous_trading_date)
        sorted_data['prev_date'] = pd.to_datetime(sorted_data['prev_date'])
        sorted_data = add_factor(sorted_data, 'market_cap_3_ma3',
                                  date_field='prev_date', asset_field='order_book_id')

        # 阈值来自配置
        filter_sorted_df = sorted_data[
            (sorted_data['total_turnover_ma10'] >= self._rt_min_turnover) &
            (sorted_data['market_cap_3_ma3'] >= self._rt_min_market_cap)
        ].reset_index(drop=True)

        # 选股范围
        if sample_range:
            sr_df = self._client.get_stock_index_comments(
                sample_range, start=self._pre_date, end=self._pre_date).T
            sr_codes = sr_df[sr_df[self._pre_date] == 1].index.tolist()
            filter_sorted_df = filter_sorted_df[
                filter_sorted_df['order_book_id'].isin(sr_codes)].reset_index(drop=True)

        filter_sorted_df = filter_sorted_df.sort_values(
            by=self._rt_score_rename, ascending=False)
        filter_sorted_codes = filter_sorted_df['order_book_id'].unique().tolist()

        # 行业
        industries = rqdatac.get_instrument_industry(
            order_book_ids=filter_sorted_codes, date=self._pre_date
        )['first_industry_name'].reset_index().rename(
            columns={'first_industry_name': 'industry'})
        filter_sorted_df = pd.merge(
            filter_sorted_df,
            industries[['order_book_id', 'industry']],
            on='order_book_id', how='left')

        # 剔除停牌
        stop_info = rqdatac.is_suspended(
            filter_sorted_codes, start_date=self._date,
            end_date=self._date, market='cn').T
        stop_info.columns = ['stop']
        stop_codes = stop_info[stop_info['stop'] == True].index.tolist()
        if stop_codes:
            self._logger.debug(f'  剔除停牌: {len(stop_codes)}只  {stop_codes[:5]}...')
        filter_sorted_df = filter_sorted_df[
            ~filter_sorted_df['order_book_id'].isin(stop_codes)].reset_index(drop=True)

        # 剔除 ST
        st_info = rqdatac.is_st_stock(filter_sorted_codes, self._date, self._date).T
        st_info.columns = ['is_st']
        st_codes = st_info[st_info['is_st'] == True].index.tolist()
        if st_codes:
            self._logger.debug(f'  剔除ST: {len(st_codes)}只  {st_codes[:5]}...')
        filter_sorted_df = filter_sorted_df[
            ~filter_sorted_df['order_book_id'].isin(st_codes)].reset_index(drop=True)

        self._logger.info(f'  候选池: {len(filter_sorted_df)} 只')
        return filter_sorted_df[[
            'order_book_id', 'date', self._rt_score_rename, 'industry']]

    # ====================================================================
    # 上期目标权重
    # ====================================================================
    def _get_pre_target_weights(self, bench_mark, sample_range, unique_id):
        pre_path = os.path.join(
            self.target_path, self.pre_date,
            f'{self.pre_date}_{unique_id}_{bench_mark}_{sample_range}_target.csv')
        if os.path.exists(pre_path):
            pre = pd.read_csv(pre_path)
            pre['code'] = pre['code'].apply(lambda x: int(str(x)[:6]))
            pre['symbol'] = pre['code'].apply(lambda x: str(x).zfill(6))
            pre['exchange'] = pre['symbol'].apply(
                lambda x: 'XSHG' if x[:1] == '6' else 'XSHE')
            pre['symbol'] = pre['symbol'] + '.' + pre['exchange']
            pre = pre.set_index('symbol')
            weights = pre['w'].to_dict()
            self._logger.debug(f'  上期权重: {len(weights)} 只')
            return weights
        return None

    # ====================================================================
    # 组合优化 → 生成 target
    # ====================================================================
    def _calculate_target(self, unique_id, sample_range, bench_mark,
                           dealed_model_data, benchmark_industry_weights,
                           max_stock_weight, turnover_limit,
                           min_industry_limit, max_industry_limit,
                           pre_target_weights, parameter,
                           stock_factors, index_exposures,
                           index_weight_limits=None, direction=1):
        raw_target = portfolio_optimizer.main(
            parameter, dealed_model_data, pre_target_weights,
            style_df=stock_factors, bench_style_df=index_exposures,
            index_weight_limits=index_weight_limits)

        raw_target = raw_target.sort_values(
            by=self._rt_score_rename, ascending=False).reset_index()

        target = (raw_target[['order_book_id', 'weight']]
                  .rename(columns={'order_book_id': 'code', 'weight': 'w'})
                  .sort_values(by='w', ascending=False)
                  .reset_index(drop=True))

        # 一行摘要
        selected_df = raw_target.copy()
        selected_df['sum'] = (selected_df[self._rt_score_rename]
                              * selected_df['weight'])
        total_score = selected_df.groupby('industry')['sum'].sum().sum()
        top3 = ', '.join(
            f'{r.code}({r.w:.3f})'
            for r in target.head(3).itertuples(index=False))
        self._logger.info(
            f'  优化结果: {len(selected_df)}只 | 目标值={total_score:.4f} | Top3: {top3}')

        # 行业分布 → DEBUG
        iw = selected_df.groupby('industry')['weight'].sum().sort_values(ascending=False)
        self._logger.debug(f'  行业分布:\n{iw.to_string()}')

        # 保存
        target_dir = os.path.join(self.target_path, self.date)
        os.makedirs(target_dir, exist_ok=True)
        target_file = os.path.join(
            target_dir,
            f'{self.date}_{unique_id}_{bench_mark}_{sample_range}_target.csv')
        target.to_csv(target_file, encoding='gbk')
        self._logger.debug(f'  已保存: {target_file}')

    # ====================================================================
    # 处理单个 unique_id
    # ====================================================================
    def _process_unique_id(self, unique_id, parameter):
        bench_mark = parameter.get('bench_mark')
        sample_range = parameter.get('sample_range')
        max_stock_w = parameter.get('max_stock_weight')
        turnover = parameter.get('turnover_limit')
        min_ind = parameter.get('min_industry')
        max_ind = parameter.get('max_industry')
        black_list = parameter.get('black_list')
        forbid_lst = parameter.get('forbid_lst', [])
        sw_mult = parameter.get('stock_weight_multiplier', 1.1)
        sw_fb = parameter.get(
            'stock_weight_fallback',
            max_stock_w if max_stock_w is not None else 0.02)

        biw = self._get_benchmark_industry_weights(bench_mark)
        dmd = self._get_model_data_and_deal(
            black_list, forbid_lst, sample_range=sample_range)
        ptw = self._get_pre_target_weights(bench_mark, sample_range, unique_id)

        iwl = self._get_index_weight_limits(
            sample_range=sample_range,
            bench_mark=bench_mark,
            stock_weight_multiplier=sw_mult,
            stock_weight_fallback=sw_fb)

        self._calculate_target(
            unique_id, sample_range, bench_mark, dmd,
            biw, max_stock_w, turnover,
            min_ind, max_ind, ptw, parameter,
            self.stock_factors, self.index_exposures,
            index_weight_limits=iwl, direction=1)

    # ====================================================================
    # 主入口
    # ====================================================================
    def main(self):
        total = len(self.parameters)
        self._logger.info(f'{"=" * 55}')
        self._logger.info(f'开始处理 {total} 个 target')
        self._logger.info(f'{"=" * 55}')

        results = []
        for idx, (uid, param) in enumerate(self.parameters.items(), 1):
            bench = param.get('bench_mark', '?')
            bw = param.get('max_stock_weight', '?')
            self._logger.info(f'[{idx}/{total}] {uid}  bench={bench}  max_w={bw}')
            try:
                self._process_unique_id(uid, param)
                results.append((uid, True, ''))
            except Exception as e:
                err_msg = str(e)[:80]
                self._logger.error(f'  ✗ 失败: {err_msg}')
                results.append((uid, False, err_msg))

        ok = sum(1 for _, s, _ in results if s)
        fail = total - ok
        self._logger.info(f'{"=" * 55}')
        self._logger.info(f'处理汇总: 成功 {ok}/{total}  |  失败 {fail}/{total}')
        for uid, status, err in results:
            mark = '✓' if status else '✗'
            line = f'  [{mark}] {uid}'
            if err:
                line += f'  —  {err}'
            self._logger.info(line)
        self._logger.info(f'{"=" * 55}')

    def main1(self, uid):
        """处理单个 unique_id（调试用）"""
        param = self.parameters[uid]
        self._process_unique_id(uid, param)


# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    print(f'根路径: {root_path}')
    config = Config(f'{root_path}/target_config.yaml')

    log_cfg = config.get('logging', {})
    logger = setup_logger(
        name='tranform_target',
        log_dir=log_cfg.get('log_dir'),
        level=log_cfg.get('level', 'INFO'),
        max_bytes=log_cfg.get('max_bytes', 10 * 1024 * 1024),
        backup_count=log_cfg.get('backup_count', 30),
    )

    rq_cfg = config.get('rqdatac', {})
    rqdatac.init(
        username=rq_cfg.get('username', 'license'),
        password=rq_cfg.get('password', ''),
        use_pool=rq_cfg.get('use_pool', True),
        max_pool_size=rq_cfg.get('max_pool_size', 8),
    )
    logger.info('rqdatac 初始化完成')

    date = sys.argv[1] if len(sys.argv) > 1 else config.get('date')
    logger.info(f'tranform_target 启动 | 日期={date}')

    tt = TransformTarget(config, date)
    tt.main()

    logger.info('tranform_target 全部完成')
