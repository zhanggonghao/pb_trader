"""
米筐数据获取模块 - 获取股票价格、因子暴露等数据
"""
import config
import pandas as pd
import numpy as np


def normalize_date_for_rq(date_str):
    """将YYYYMMDD转换为米筐需要的YYYY-MM-DD格式"""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str



def get_futures_contract_codes(base_date_str):
    """
    根据基准日期自动计算股指期货合约代码
    
    中金所股指期货（IF/IC/IM）挂牌规则：当月、下月、随后两个季月
    季月循环：3月、6月、9月、12月
    
    参数:
        base_date_str: 基准日期 YYYYMMDD（通常是报告期最后一天）
    
    返回:
        dict: {
            'near': [IF代码, IC代码, IM代码],   # 当月合约
            'far':  [IF代码, IC代码, IM代码],   # 最远季月合约
        }
    """
    from datetime import datetime
    
    dt = datetime.strptime(base_date_str, '%Y%m%d')
    year, month = dt.year, dt.month
    
    quarterly = [3, 6, 9, 12]
    
    # 当月合约 month1
    # 判断当月合约是否已到期（第三个星期五）
    # 到期后使用下月合约作为近月，否则当月合约无法获取数据
    import calendar
    _first_day = datetime(year, month, 1)
    _days_to_fri = (4 - _first_day.weekday()) % 7
    _third_friday = _first_day + __import__('datetime').timedelta(days=_days_to_fri + 14)
    if dt > _third_friday:
        if month == 12:
            _near_month = 1
            _near_year = year + 1
        else:
            _near_month = month + 1
            _near_year = year
    else:
        _near_month = month
        _near_year = year
    

    y1, m1 = _near_year, _near_month

    # 下月合约 month2
    if _near_month == 12:
        y2, m2 = _near_year + 1, 1
    else:
        y2, m2 = _near_year, _near_month + 1

    # 第一个季月（>= 下月）month3
    for q in quarterly:
        if q > m2:
            y3, m3 = y2, q
            break
    else:
        y3, m3 = y2 + 1, quarterly[0]

    # 第二个季月（最远月）month4
    idx = quarterly.index(m3)
    if idx + 1 < len(quarterly):
        y4, m4 = y3, quarterly[idx + 1]
    else:
        y4, m4 = y3 + 1, quarterly[0]

    def _make_codes(y, m):
        suffix = f'{y % 100:02d}{m:02d}'
        return [f'IF{suffix}', f'IC{suffix}', f'IM{suffix}']

    return {
        'near': _make_codes(y1, m1),
        'far':  _make_codes(y4, m4),
    }


class RQDataFetcher:
    def __init__(self):
        self.connected = False
        self.client = None

    def connect(self):
        """连接米筐，尝试多个密码"""
        if self.connected:
            return True

        for pwd in config.RQ_PASSWORDS:
            try:
                import rqdatac
                rqdatac.init(config.RQ_USERNAME, pwd)
                self.connected = True
                self.client = rqdatac
                print(f"米筐连接成功")
                return True
            except Exception as e:
                print(f"米筐连接失败: {e}")
                continue
        return False

    def reconnect(self):
        """重新连接"""
        self.connected = False
        self.client = None
        return self.connect()

    def get_benchmark_prices(self, dates):
        """获取基准指数每日收盘价"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            start_date = normalize_date_for_rq(dates[0])
            end_date = normalize_date_for_rq(dates[-1])

            df = self.client.get_price(
                config.BENCHMARK_CODE,
                start_date=start_date,
                end_date=end_date,
                fields=['close']
            )

            if df is None or df.empty:
                return None

            result = {}
            for idx in df.index:
                date_obj = idx[1] if isinstance(idx, tuple) else idx
                if hasattr(date_obj, 'strftime'):
                    result[date_obj.strftime('%Y%m%d')] = float(df.loc[idx, 'close'])
                else:
                    result[str(date_obj).replace('-', '')] = float(df.loc[idx, 'close'])

            return result

        except Exception as e:
            print(f"获取基准价格失败: {e}")
            self.reconnect()
            return None

    def get_stock_prices(self, stock_codes, dates):
        """获取股票收盘价"""
        if not self.connected:
            if not self.connect():
                return None

        if not stock_codes:
            return {}

        try:
            start_date = normalize_date_for_rq(dates[0])
            end_date = normalize_date_for_rq(dates[-1])

            valid_codes = [c for c in stock_codes if c and isinstance(c, str) and '.' in c]

            if not valid_codes:
                return {}

            df = self.client.get_price(
                valid_codes,
                start_date=start_date,
                end_date=end_date,
                fields=['close'],
                adjust_type='pre'
            )

            if df is None or df.empty:
                return {}

            result = {}
            for code in valid_codes:
                result[code] = {}

            if hasattr(df, 'columns') and 'close' in df.columns:
                series = df['close']
            else:
                series = df

            for idx in series.index:
                code = idx[0] if isinstance(idx, tuple) else idx
                date_obj = idx[1] if isinstance(idx, tuple) else idx
                if hasattr(date_obj, 'strftime'):
                    date_str = date_obj.strftime('%Y%m%d')
                else:
                    date_str = str(date_obj).replace('-', '')

                if code in result:
                    try:
                        result[code][date_str] = float(series.loc[idx])
                    except:
                        pass

            return result

        except Exception as e:
            print(f"获取股票价格失败: {e}")
            self.reconnect()
            return {}

    def get_style_factor_exposure(self, stock_codes, date_str):
        """获取风格因子暴露"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            start_date = normalize_date_for_rq(date_str)
            end_date = normalize_date_for_rq(date_str)

            valid_codes = [c for c in stock_codes if c and isinstance(c, str) and '.' in c]

            if not valid_codes:
                return None

            df = self.client.get_style_factor_exposure(
                valid_codes,
                start_date=start_date,
                end_date=end_date
            )
            if df is None or df.empty:
                return None

            result = {}
            for factor in ['size', 'non_linear_size', 'momentum', 'liquidity',
                          'book_to_price', 'leverage', 'growth', 'earnings_yield',
                          'beta', 'residual_volatility']:
                if factor in df.columns:
                    result[factor] = {}
                    for idx in df.index:
                        code = idx[0] if isinstance(idx, tuple) else idx
                        try:
                            result[factor][code] = float(df.loc[idx, factor])
                        except:
                            pass
            return result
        

        except Exception as e:
            print(f"获取风格因子失败: {e}")
            self.reconnect()
            return None

    def get_industry_classification(self, stock_codes, date_str):
        """获取申万一级行业分类"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            rq_date = normalize_date_for_rq(date_str)
            valid_codes = [c for c in stock_codes if c and isinstance(c, str) and '.' in c]

            if not valid_codes:
                return None

            df = self.client.shenwan_instrument_industry(
                valid_codes,
                date=rq_date,
                level=1,
                expect_df=True
            )

            if df is None or df.empty:
                return None

            result = {}
            for code in df.index:
                # 使用行业名称作为分类依据
                if 'index_name' in df.columns:
                    result[code] = str(df.loc[code, 'index_name'])
                elif 'index_code' in df.columns:
                    result[code] = str(df.loc[code, 'index_code'])
                else:
                    result[code] = str(df.iloc[0, 0])

            return result

        except Exception as e:
            print(f"获取行业分类失败: {e}")
            self.reconnect()
            return None

    def get_benchmark_constituents(self, date_str):
        """获取沪深300指数成分股"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            const_date = normalize_date_for_rq(date_str)
            df = self.client.index_weights_ex(
                config.BENCHMARK_CODE,
                start_date=const_date, end_date=const_date
            )

            if df is None or df.empty:
                return None

            return df.index.get_level_values(1).tolist()

        except Exception as e:
            print(f"获取指数成分失败: {e}")

            try:
                codes = self.client.all_instruments(
                    type='CS',
                    date=normalize_date_for_rq(date_str)
                )

                if codes is None or codes.empty:
                    return None

                return codes['order_book_id'].tolist()[:300]

            except Exception as e2:
                print(f"备用获取指数成分也失败: {e2}")
                self.reconnect()
                return None

    def get_benchmark_factor_exposure(self, date_str):
        """获取基准指数的风格因子暴露（采样）"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            codes = self.get_benchmark_constituents(date_str)
            if not codes:
                return None

            sample_codes = codes[:100]

            sf = self.get_style_factor_exposure(sample_codes, date_str)
            if not sf:
                return None

            result = {}
            for factor in ['size', 'non_linear_size', 'momentum', 'liquidity',
                          'book_to_price', 'leverage', 'growth', 'earnings_yield',
                          'beta', 'residual_volatility']:
                if factor in sf and sf[factor]:
                    vals = [v for v in sf[factor].values() if v is not None and not np.isnan(v)]
                    result[factor] = np.mean(vals) if vals else 0
                else:
                    result[factor] = 0

            return result

        except Exception as e:
            print(f"获取基准因子暴露失败: {e}")
            self.reconnect()
            return None

    def get_previous_trading_date(self, date_str):
        """获取指定日期的前一个交易日"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            rq_date = normalize_date_for_rq(date_str)
            prev_date = self.client.get_previous_trading_date(rq_date)
            if hasattr(prev_date, 'strftime'):
                return prev_date.strftime('%Y%m%d')
            return str(prev_date).replace('-', '')
        except Exception as e:
            print(f"获取前一交易日失败: {e}")
            return None

    def get_benchmark_industry_weights(self, date_str):
        """获取基准指数的申万一级行业权重分布"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            codes = self.get_benchmark_constituents(date_str)
            if not codes:
                return None

            # 获取成分股的行业分类
            industry_map = self.get_industry_classification(codes, date_str)
            if not industry_map:
                return None

            # 获取成分股权重
            const_date = normalize_date_for_rq(date_str)
            weight_df = self.client.index_weights_ex(
                config.BENCHMARK_CODE,
                start_date=const_date, end_date=const_date
            )

            if weight_df is None or weight_df.empty:
                # 如果获取不到权重，使用等权近似
                industry_weights = {}
                for code, industry in industry_map.items():
                    if industry:
                        industry_weights[industry] = industry_weights.get(industry, 0) + 1.0
                total = sum(industry_weights.values())
                if total > 0:
                    for k in industry_weights:
                        industry_weights[k] /= total
                return industry_weights

            # 用实际权重汇总行业
            industry_weights = {}
            for idx in weight_df.index:
                code = idx[1] if isinstance(idx, tuple) else idx
                weight = float(weight_df.loc[idx, 'weight']) if 'weight' in weight_df.columns else 0
                industry = industry_map.get(code)
                if industry:
                    industry_weights[industry] = industry_weights.get(industry, 0) + weight

            return industry_weights

        except Exception as e:
            print(f"获取基准行业权重失败: {e}")
            self.reconnect()
            return None

    def _last_date_str(self, dates):
        """从日期列表中取最后一个交易日"""
        if not dates:
            from datetime import datetime
            return datetime.now().strftime('%Y%m%d')
        return dates[-1]

    def get_market_overview_data(self, dates):
        """
        获取市场宏观数据（用于市场回顾文字）
        
        参数:
            dates: 日期列表，包含前一交易日和报告周期所有日期
        
        返回:
            dict: 包含各指数价格、期货价格、成交额等数据
        """
        if not self.connected:
            if not self.connect():
                return None

        try:
            start_date = normalize_date_for_rq(dates[0])
            end_date = normalize_date_for_rq(dates[-1])

            result = {}

            # 1. 上证指数
            try:
                df_sh = self.client.get_price(
                    '000001.XSHG',
                    start_date=start_date, end_date=end_date,
                    fields=['close', 'volume', 'total_turnover']
                )
                if df_sh is not None and not df_sh.empty:
                    result['sh_index'] = {}
                    for idx in df_sh.index:
                        date_obj = idx[1] if isinstance(idx, tuple) else idx
                        date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                        result['sh_index'][date_str] = {
                            'close': float(df_sh.loc[idx, 'close']),
                            'volume': float(df_sh.loc[idx, 'volume']),
                            'turnover': float(df_sh.loc[idx, 'total_turnover']),
                        }
            except Exception as e:
                print(f"获取上证指数失败: {e}")

            # 2. 沪深300
            try:
                df_hs300 = self.client.get_price(
                    '000300.XSHG',
                    start_date=start_date, end_date=end_date,
                    fields=['close']
                )
                if df_hs300 is not None and not df_hs300.empty:
                    result['hs300'] = {}
                    for idx in df_hs300.index:
                        date_obj = idx[1] if isinstance(idx, tuple) else idx
                        date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                        result['hs300'][date_str] = float(df_hs300.loc[idx, 'close'])
            except Exception as e:
                print(f"获取沪深300失败: {e}")

            # 3. 中证500
            try:
                df_zz500 = self.client.get_price(
                    '000905.XSHG',
                    start_date=start_date, end_date=end_date,
                    fields=['close']
                )
                if df_zz500 is not None and not df_zz500.empty:
                    result['zz500'] = {}
                    for idx in df_zz500.index:
                        date_obj = idx[1] if isinstance(idx, tuple) else idx
                        date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                        result['zz500'][date_str] = float(df_zz500.loc[idx, 'close'])
            except Exception as e:
                print(f"获取中证500失败: {e}")

            # 4. 中证1000
            try:
                df_zz1000 = self.client.get_price(
                    '000852.XSHG',
                    start_date=start_date, end_date=end_date,
                    fields=['close']
                )
                if df_zz1000 is not None and not df_zz1000.empty:
                    result['zz1000'] = {}
                    for idx in df_zz1000.index:
                        date_obj = idx[1] if isinstance(idx, tuple) else idx
                        date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                        result['zz1000'][date_str] = float(df_zz1000.loc[idx, 'close'])
            except Exception as e:
                print(f"获取中证1000失败: {e}")

            # 5. 创业板指
            try:
                df_cyb = self.client.get_price(
                    '399006.XSHE',
                    start_date=start_date, end_date=end_date,
                    fields=['close']
                )
                if df_cyb is not None and not df_cyb.empty:
                    result['cyb'] = {}
                    for idx in df_cyb.index:
                        date_obj = idx[1] if isinstance(idx, tuple) else idx
                        date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                        result['cyb'][date_str] = float(df_cyb.loc[idx, 'close'])
            except Exception as e:
                print(f"获取创业板指失败: {e}")

            # 6. 股指期货（自动获取合约代码）
            fut_codes = get_futures_contract_codes(self._last_date_str(dates))
            for label, codes in [('当月合约', fut_codes['near']), ('最远季月合约', fut_codes['far'])]:
                for fut_code in codes:
                    try:
                        df_fut = self.client.get_price(
                            fut_code,
                            start_date=start_date, end_date=end_date,
                            fields=['close']
                        )
                        if df_fut is not None and not df_fut.empty:
                            result[fut_code.lower()] = {}
                            for idx in df_fut.index:
                                date_obj = idx[1] if isinstance(idx, tuple) else idx
                                date_str = date_obj.strftime('%Y%m%d') if hasattr(date_obj, 'strftime') else str(date_obj).replace('-', '')
                                result[fut_code.lower()][date_str] = float(df_fut.loc[idx, 'close'])
                    except Exception as e:
                        print(f"获取{fut_code}({label})失败: {e}")

            return result if result else None

        except Exception as e:
            print(f"获取市场宏观数据失败: {e}")
            return None
