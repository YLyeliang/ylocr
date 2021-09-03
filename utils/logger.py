# -*- coding: utf-8 -*-
# @Time : 2021/8/31 上午9:27
# @Author: yl
# @File: logger.py

import logging
import os
import time


def initLogger(log_root):
    logger = logging.getLogger("ocr")
    stream_hander = logging.StreamHandler()
    handlers = [stream_hander]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_handler = logging.FileHandler(f"{os.path.join(log_root, timestamp)}.log", 'w')
    handlers.append(file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
