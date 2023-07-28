import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # set logger level

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
