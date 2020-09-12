import starry
import numpy as np
import matplotlib.pyplot as plt
from starry.maps import logger
from theano.compile.debugmode import DebugMode

starry.config.mode = DebugMode(check_isfinite=False)

ydeg = 15

"""
logger.info("1")
map = starry.Map(ydeg, lazy=True)
logger.info("2")
map.load("earth")
logger.info("3")
x = map.render(projection="moll").eval()
logger.info(x)
logger.info("4")
"""

logger.info("1")
map = starry.Map(ydeg, lazy=False)
logger.info("2")
map.load("earth")
logger.info("3")
x = map.render(projection="moll")
logger.info(x)
logger.info("4")
