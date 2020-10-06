# Version (SCM)
from .starry_process_version import __version__

# Allow C code caching even in dev mode?
CACHE_DEV_C_CODE = True

# Set up the logger
import logging

logger = logging.getLogger("starry_process")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Main imports
from . import (
    math,
    design,
    sp,
    ops,
    integrals,
    latitude,
    size,
    longitude,
    wigner,
    transforms,
    defaults,
)
from .sp import StarryProcess
