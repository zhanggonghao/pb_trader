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
        """获取行业分类"""
        if not self.connected:
            if not self.connect():
                return None

        try:
            start_date = normalize_date_for_rq(date_str)

            valid_codes = [c for c in stock_codes if c and isinstance(c, str) and '.' in c]

            if not valid_codes:
                return None

            df = self.client.get_industry(
                valid_codes,
                date=start_date,
                source='sw'
            )

            if df is None or df.empty:
                return None

            result = {}
            for idx in df.index:
                code = idx[0] if isinstance(idx, tuple) else idx
                result[code] = str(df.loc[idx, 'industry_code'])

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
            print(const_date)
            df = self.client.index_weights_ex(
                config.BENCHMARK_CODE,
                start_date=const_date, end_date=const_date
            )
            print(df)

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
