import rqdatac # type: ignore
# rqdatac.init(13601611030, 'PB123456')
rqdatac.init(username="license", password="jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30=", use_pool=True, max_pool_size=8)


df = rqdatac.get_price(order_book_ids='000012.XSHG', frequency='1d', start_date='2013-07-26', end_date='2013-07-30', adjust_type='post', expect_df=True)

print(df)

df = rqdatac.get_price(order_book_ids='000012.XSHG', frequency='5m', start_date='2013-07-26', end_date='2013-07-30', adjust_type='post', expect_df=True)

print(df)