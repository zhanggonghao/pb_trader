
import rqdatac # type: ignore
# rqdatac.init(13601611030, 'PB123456789')
rqdatac.init(username="license", password="gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc=", use_pool=True, max_pool_size=8)

# rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)


from .stock_basics import (download_st_stock_info, download_suspended_stock_info, update_stocks_basic_info,
                           update_index_components, update_index_components_weights, update_index_components_weights_industry,
                           update_all_instruments_industry)

from .stock_price import (update_stock_daily_prices, update_stock_minute_prices_parallel, update_stock_daily_vwap_prices,
                          update_stock_daily_twap_prices, update_stock_vwap_twap_xmin_prices, extend_index1000_prices_by_index500)

from .future_price import (update_futures_minute_prices, update_futures_state_matrix, update_futures_consistent_prices)

from .stock_factor import (update_daily_simple_factors, update_daily_compound_factors, update_compound_factor_data_xmin, combine_index_related_factors)

from .future_data import (update_future_data,  update_option_data, update_tick_data, update_benchmark_data)

# from .rqfactors import update_rqfactors_daily