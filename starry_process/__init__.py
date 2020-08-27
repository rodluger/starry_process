# Set up the logger
import logging

logger = logging.getLogger("starry_process")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Force double precision
import theano.tensor as tt

tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"
del tt

# Main imports
from .starry_process_version import __version__
from .gp import YlmGP
from . import (
    gp,
    math,
    ops,
    integrals,
    latitude,
    size,
    longitude,
    contrast,
    wigner,
    transforms,
)

