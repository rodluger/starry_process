# Version (SCM)
from .starry_process_version import __version__

# Set up the logger
import logging

logger = logging.getLogger("starry_process")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Main imports
from . import (
    math,
    sp,
    ops,
    integrals,
    latitude,
    size,
    longitude,
    contrast,
    wigner,
    transforms,
)
from .sp import StarryProcess
