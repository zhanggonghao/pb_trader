"""
共享配置加载模块
统一 YAML 配置文件的读取与类型化访问，支持多文件合并。
"""
import os
import yaml
import datetime as dt
from typing import Any, Optional


class Config:
    """配置容器，封装 dict 并提供 .get() / [] 访问，兼容原有 config.get(key) 用法。"""

    def __init__(self, *yaml_paths: str):
        """
        按顺序加载多个 YAML 文件，后加载的键覆盖先加载的（浅合并）。

        Args:
            *yaml_paths: 一个或多个 YAML 文件路径
        """
        self._data: dict = {}
        for path in yaml_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        self._data.update(loaded)

        # 自动解析日期：若 date == 'current'，替换为当日
        date_str = self._data.get('date', 'current')
        if date_str == 'current':
            self._data['date'] = dt.datetime.now().strftime('%Y%m%d')

    # ---- 与普通 dict 兼容的接口 ----
    def get(self, key: str, default: Any = None) -> Any:
        """获取顶层键的值。兼容原有 config.get(key, default) 写法。"""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    # ---- 嵌套访问 ----
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        支持 '.' 分隔的嵌套键访问。

        Example:
            config.get_nested('rqdatac.username')
        """
        keys = key_path.split('.')
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    # ---- 工具方法 ----
    def get_all(self) -> dict:
        """返回内部 dict 的引用（谨慎修改）。"""
        return self._data

    def resolve_date(self, cli_date: Optional[str] = None) -> str:
        """
        解析最终运行日期：命令行参数 > 配置文件 date 字段。

        Args:
            cli_date: 命令行传入的日期字符串 (YYYYMMDD)

        Returns:
            日期字符串 YYYYMMDD
        """
        if cli_date:
            return cli_date
        return self._data.get('date', dt.datetime.now().strftime('%Y%m%d'))


def load_config(*paths: str) -> Config:
    """快捷函数：加载一个或多个配置文件。"""
    return Config(*paths)
