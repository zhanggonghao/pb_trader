import os
import logging


# 获取当前脚本的文件名
script_name = os.path.basename(__file__)

def log_message(message, log_level=logging.INFO, script_name='nothing.py'):
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s')
    logger = logging.getLogger(script_name)
    logger.info(message)
