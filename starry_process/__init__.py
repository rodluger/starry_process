# Main imports
from .starry_process_version import __version__
from .gp import YlmGP
from . import gp, ops, integrals, latitude, size, longitude, contrast, wigner


# Force double precision
import theano.tensor as tt

tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"
del tt
