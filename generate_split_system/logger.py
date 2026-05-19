"""
日志模块
支持控制台 + 文件双输出，文件按大小自动轮转。
"""
import os
import logging
import logging.handlers
from typing import Optional


def setup_logger(
    name: str = 'split_system',
    log_dir: Optional[str] = None,
    level: str = 'INFO',
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 30,
) -> logging.Logger:
    """
    创建并配置一个 Logger 实例。

    - 控制台输出 INFO 及以上级别
    - 文件输出 DEBUG 及以上级别（自动轮转，保留最近 backup_count 个备份）

    Args:
        name:         日志器名称
        log_dir:      日志文件存放目录；为 None 则不写文件
        level:        日志级别名称，如 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR'
        max_bytes:    单个日志文件最大字节数，超出后自动轮转
        backup_count: 保留的历史日志文件数量

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（幂等）
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 统一格式
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # ---- 控制台 handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- 文件 handler（轮转） ----
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'split_system') -> logging.Logger:
    """获取已存在的 Logger；若不存在则返回一个基础 Logger。"""
    return logging.getLogger(name)
