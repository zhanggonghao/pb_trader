import pandas as pd
import os
import rqdatac # type: ignore

def download_all_trading_dates(data_path, market='cn'):
    """
    获取所有交易日序列（默认中国交易日历）
    """
    if market == 'cn':
        _start_date = '1990-12-19'
        _end_date = '2099-12-31'
    else:
        raise ValueError(f"仅支持:{market}的交易日历下载。")
    trading_dates = rqdatac.get_trading_dates(_start_date, _end_date)
    trading_dates = [date.strftime("%Y-%m-%d") for date in trading_dates]
    print(trading_dates)
    pd.to_pickle(trading_dates, os.path.join(data_path, 'stocks/basics/calendar.pickle'))
    return True


if __name__ == "__main__":
    # rqdatac.init(13601611030, 'PB123456789')
    
    rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)

    data_path = "/home/samba/Market/"
    download_all_trading_dates(data_path=data_path)
