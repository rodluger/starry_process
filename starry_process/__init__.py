from .starry_process_version import __version__
import theano.tensor as tt

# Force double precision
tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"
del tt
