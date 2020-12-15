# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/4

import logging, os

logger = logging.getLogger("root")

stream_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt='%(asctime)s\t%(levelname)s\t%(name)s '
                                      '%(filename)s:%(lineno)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s\t%(levelname)s\t%(name)s '
                           '%(filename)s:%(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def set_log_file(log_path):
    if os.path.isfile(log_path):
        r_handler = logging.FileHandler(log_path, 'a', 'utf-8')
    else:
        r_handler = logging.FileHandler(log_path, 'w', 'utf-8')
    r_handler.setFormatter(log_formatter)
    r_handler.setLevel(logging.INFO)
    logger.addHandler(r_handler)
    logging.basicConfig(level=logging.INFO,
                        handlers=[r_handler],
                        format='%(asctime)s\t%(levelname)s\t%(name)s%(filename)s:%(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

