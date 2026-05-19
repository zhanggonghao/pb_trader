"""
main.py
主编排入口 — 统一流程：生成目标权重 → 拆分交易指令

用法:
    python main.py [YYYYMMDD]              # 跑全流程
    python main.py [YYYYMMDD] --step 1     # 仅跑 Step 1（生成 target）
    python main.py [YYYYMMDD] --step 2     # 仅跑 Step 2（拆分指令）
"""
import os
import sys
import argparse
import datetime as dt
from pathlib import Path

# 将项目根目录加入搜索路径（访问 ultis 等公共模块）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rqdatac
from config_loader import Config
from logger import setup_logger, get_logger

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='配邦拆分系统 — 目标权重生成 + 交易指令拆分',
    )
    parser.add_argument(
        'date', nargs='?', default=None,
        help='运行日期 (YYYYMMDD)，不传则使用配置文件中的 date',
    )
    parser.add_argument(
        '--step', type=int, choices=[1, 2], default=0,
        help='仅执行指定步骤: 1=生成target, 2=拆分指令 (默认: 0=全流程)',
    )
    return parser.parse_args()


def init_rqdatac(config: Config, logger):
    """初始化 rqdatac 连接池（全局单次调用）。"""
    rq_cfg = config.get('rqdatac', {})
    rqdatac.init(
        username=rq_cfg.get('username', 'license'),
        password=rq_cfg.get('password', ''),
        use_pool=rq_cfg.get('use_pool', True),
        max_pool_size=rq_cfg.get('max_pool_size', 8),
    )
    logger.info("rqdatac 连接池初始化完成")


def step1_generate_target(date: str, logger):
    """
    Step 1 — 生成目标权重文件
    读取模型预测 → 组合优化 → 写入 data/target/{date}/ 下 CSV
    """
    logger.info('=' * 60)
    logger.info(f'Step 1/2: 生成目标权重文件 (日期: {date})')
    logger.info('=' * 60)

    config = Config(f'{ROOT}/target_config.yaml')
    # 用传入日期覆盖配置文件中的 date
    config._data['date'] = date

    from tranform_target import TransformTarget
    tt = TransformTarget(config, date)
    tt.main()

    logger.info('Step 1/2 完成')


def step2_split_orders(date: str, logger):
    """
    Step 2 — 生成交易拆分指令
    读取 Step 1 生成的 target CSV + 昨收持仓 → 输出调仓/建仓/T0 指令
    """
    logger.info('=' * 60)
    logger.info(f'Step 2/2: 生成交易拆分指令 (日期: {date})')
    logger.info('=' * 60)

    config = Config(f'{ROOT}/split_system.yaml')
    config._data['date'] = date

    from split_system import SplitSystem
    ss = SplitSystem(date, config)
    ss.main()

    logger.info('Step 2/2 完成')


def main():
    args = parse_args()

    # ---- 日期解析 ----
    # 优先加载 split_system 配置获取默认日期（两套配置的默认日期应该一致）
    config = Config(f'{ROOT}/split_system.yaml')
    date = args.date or config.get('date')
    if date == 'current':
        date = dt.datetime.now().strftime('%Y%m%d')

    # ---- 日志初始化 ----
    log_cfg = config.get('logging', {})
    logger = setup_logger(
        name='main',
        log_dir=log_cfg.get('log_dir'),
        level=log_cfg.get('level', 'INFO'),
        max_bytes=log_cfg.get('max_bytes', 10 * 1024 * 1024),
        backup_count=log_cfg.get('backup_count', 30),
    )

    logger.info(f'{"=" * 60}')
    logger.info(f'配邦拆分系统启动 — 日期: {date}, 步骤: {"全流程" if args.step == 0 else f"仅 Step {args.step}"}')
    logger.info(f'{"=" * 60}')

    # ---- 初始化 rqdatac（全局一次） ----
    init_rqdatac(config, logger)

    exit_code = 0

    try:
        if args.step in (0, 1):
            step1_generate_target(date, logger)

        if args.step in (0, 2):
            step2_split_orders(date, logger)

    except Exception as e:
        logger.error(f'执行失败: {e}', exc_info=True)
        exit_code = 1

    logger.info(f'{"=" * 60}')
    logger.info(f'配邦拆分系统结束 — 退出码: {exit_code}')
    logger.info(f'{"=" * 60}')

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
