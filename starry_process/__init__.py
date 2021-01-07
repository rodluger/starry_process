# Version (SCM)
from .starry_process_version import __version__

# Allow C code caching even in dev mode?
CACHE_DEV_C_CODE = False

# Set up the logger
import logging

logger = logging.getLogger("starry_process")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Main imports
from . import (
    math,
    flux,
    sp,
    ops,
    integrals,
    latitude,
    size,
    longitude,
    wigner,
    defaults,
    temporal,
    visualize,
)
from .sp import StarryProcess
from .latitude import gauss2beta, beta2gauss
from .temporal import *
