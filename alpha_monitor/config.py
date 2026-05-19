"""
配置中心 - 所有可调参数集中管理
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RQDataConfig:
    """米筐数据源配置"""
    username: str = "license"
    # password: str = "gCKbHurs4dlMyehGC3GVBEYgFsPRZZiVNUWfCJCS9ifEdXYWnBqgopXvtwMg3GdeJxvb02yljxgaEYxhu1pREMs6k4oFmIU5e0Lf4k56THXNJdgY9i90ehi9i_Hh9sDDSHYg3WgNslsvOwIo4Ku66nV2P1T69RprXP0OIqsep3M=F1112RCtTHbSGqqSJUDAyNXbGm-ik0mkYJGwcAKsg8YNX6oj6u_dAnCo2tUYJ6jp7PAtYxCA3p3SXDA5xa4f_X-eZA5T2vbtFqWkHU5QEz6gDnIsCHX5JSkzUIPqToU8rLOD8D3q-MAJICrCnZ8B4y3Hp6X6KCSR_8X8vMddDkc="
    password: str = "jUrRi5rWOK6uHreZ4wu0xKpFZjBEixs5oNQWutfnMJPpZRx1Gl0tXIJ10-EXkrgE5rIkTzM64U53dN1ZPVvOe8icNOsmwUlD4lsGp5BF9zsNIhJdPIsQGUS7lHz34DID1myOgeNFKHQ09d1Ksl6uEIEx9_9k8t47PyBdAKP_4Eg=Jx6_6AXjiwzgXLUaIbCiNSUjxHL6UStZcJpDfAThNGIH-GijxfIXSBF9SQBGeerCtxJnwW1WRl47cINvGdy4X895G54jfUsMOQCeT8PO4n_TY3vWlzp8jmNcViOCgx2iqHfMlDCdCGMZ9UsSd1XEju90XNLT1gBzpDPOsaC9a30="
    max_pool_size: int = 20


@dataclass
class BenchmarkConfig:
    """基准指数配置"""
    code: str = "000300.XSHG"
    name: str = "沪深300"


@dataclass
class FutureProductConfig:
    """股指期货品种"""
    name: str = "IF"
    spot_index: str = "000300.XSHG"
    point_value: float = 300.0
    contracts: list = None


@dataclass
class AccountConfig:
    """账户配置"""
    name: str = ""
    short_name: str = ""
    qmt_dir: str = ""
    account_type: str = "stock"
    account_id: str = ""
    target_file_path: str = ""
    deposit: float = 0
    net_value_source: str = "file"
    net_value_file_pattern: str = "email/资产净值公告_*_{name}_*.xlsx"


@dataclass
class ServerConfig:
    """Web服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    refresh_interval: int = 60


@dataclass
class AppConfig:
    """总配置"""
    rqdatac: RQDataConfig = field(default_factory=RQDataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    futures: list = field(default_factory=lambda: [
        FutureProductConfig(name="IF", spot_index="000300.XSHG", point_value=300,
                            contracts=["IF2506", "IF2509", "IF2512", "IF2603"]),
        FutureProductConfig(name="IH", spot_index="000016.XSHG", point_value=300,
                            contracts=["IH2506", "IH2509", "IH2512", "IH2603"]),
        FutureProductConfig(name="IC", spot_index="000905.XSHG", point_value=200,
                            contracts=["IC2506", "IC2509", "IC2512", "IC2603"]),
        FutureProductConfig(name="IM", spot_index="000852.XSHG", point_value=200,
                            contracts=["IM2506", "IM2509", "IM2512", "IM2603"]),
    ])

    accounts: list = field(default_factory=lambda: [
        AccountConfig(
            name="海通PBZS1H",
            short_name="PBZS1H",
            qmt_dir=r"E:\qmt_auto_export\SHPB0649",
            account_type="stock",
            account_id="132803",
            target_file_path=r"E:\code\DailyScripts\TradeData\target\{date}\{date}_TCHMD_000300.XSHG_zz1800_target.csv",
            deposit=0,
            net_value_source="file",
            net_value_file_pattern="email/资产净值公告_*_配邦中圣私募证券投资基金_*.xlsx"
        ),
        AccountConfig(
            name="海通PBHSZX1H",
            short_name="PBHSZX1H_pb",
            qmt_dir=r"E:\qmt_auto_export\SHPB0649",
            account_type="credit",
            account_id="60010146",
            target_file_path=r"E:\code\DailyScripts\TradeData\target\{date}\{date}_TCHMD_000300.XSHG_zz1800_target.csv",
            deposit=0, # 出金金额为正，如今金额为负
            net_value_source="file",
            net_value_file_pattern="email/【基金净值】SJE581_配邦投资二号私募证券投资基金_*.xlsx"
        ),
        AccountConfig(
            name="国泰海通PBHSZX1H",
            short_name="PBHSZX1H_czq",
            qmt_dir=r"E:\qmt_auto_export\SHPB0649",
            account_type="stock",
            account_id="10510783",
            target_file_path=r"E:\code\DailyScripts\TradeData\target\{date}\{date}_CZQZX300_000300.XSHG_000906.XSHG_target.csv",
            deposit=0,
            net_value_source="file",
            net_value_file_pattern="email/【基金净值】SJE581_配邦投资二号私募证券投资基金_*.xlsx"
        ),
    ])


CONFIG = AppConfig()
