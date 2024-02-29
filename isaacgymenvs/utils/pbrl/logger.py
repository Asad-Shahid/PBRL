import time
import logging

import numpy as np
import colorlog
# "%(log_color)s[%(asctime)s] %(message)s"
LogFormat = "  %(reset)s%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s"
formatter = colorlog.ColoredFormatter(
    LogFormat,
    datefmt=None,
    reset=True,
    log_colors={'DEBUG': 'cyan', 'INFO': 'white', 'WARNING': 'yellow', 'ERROR': 'red,bold', 'CRITICAL': 'red,bg_white'},
    secondary_log_colors={},
    style='%')

logger = colorlog.getLogger('gear_assembly')
logger.setLevel(logging.DEBUG)

#fh = logging.FileHandler('log')
#fh.setLevel(logging.DEBUG)
#fh.setFormatter(formatter)
#logger.addHandler(fh)

ch = colorlog.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
