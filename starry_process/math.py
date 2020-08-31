from . import ops
import numpy as np
import scipy
import theano.tensor as tt
from theano.ifelse import ifelse
from theano.tensor import nlinalg as nla
from theano.tensor import slinalg as sla


# Numpy mock RandomStreams class, no_default_updates=True
class _RandomStreams(object):
    def __init__(self, seed=0):
        self.seed(seed)

    def normal(self, size):
        res = self.random.normal(size=size)
        self.seed()
        return res

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        self.random = np.random.default_rng(self._seed)


# Force double precision in Theano
tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"


# Theano Cholesky solve
_solve_lower = tt.slinalg.Solve(A_structure="lower_triangular", lower=True)
_solve_upper = tt.slinalg.Solve(A_structure="upper_triangular", lower=False)


# PI
_PI = float(np.pi)


class MathType(type):
    """Wrapper for theano/numpy functions."""

    @property
    def ops(cls):
        if cls.use_theano:
            return ops.theano_ops
        else:
            return ops.python_ops

    def cast(cls, *args):
        if cls.use_theano:
            if len(args) == 1:
                return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
            else:
                return [
                    tt.as_tensor_variable(arg).astype(tt.config.floatX)
                    for arg in args
                ]
        else:
            if len(args) == 1:
                return np.array(args[0], dtype="float64")
            else:
                return [np.array(arg, dtype="float64") for arg in args]

    def ifelse(cls, expr, true_branch, false_branch):
        if cls.use_theano:
            return ifelse(expr, true_branch, false_branch)
        else:
            if expr:
                return true_branch
            else:
                return false_branch

    def eigen(cls, Q, neig=None, driver=None):
        """
        Returns the matrix square root of `Q`,
        computed via (hermitian) eigendecomposition:

            eigen(Q) . eigen(Q)^T = Q

        """
        if cls.use_theano:
            # TODO: Is there a way to compute only `neig` eigenvalues?
            w, U = tt.nlinalg.eigh(Q)
            U = tt.dot(U, tt.diag(tt.sqrt(tt.maximum(0, w))))
            if neig is not None:
                return U[:, -neig:]
            else:
                return U
        else:
            N = Q.shape[0]
            if neig is None or neig == N:
                kwargs = {"driver": driver}
            else:
                kwargs = {"subset_by_index": (N - neig, N - 1)}
            w, U = scipy.linalg.eigh(Q, **kwargs)
            U = U @ np.diag(np.sqrt(np.maximum(0, w)))
            return U[:, ::-1]

    def cho_solve(cls, cho_A, b):
        if cls.use_theano:
            return _solve_upper(tt.transpose(cho_A), _solve_lower(cho_A, b))
        else:
            return scipy.linalg.cho_solve((cho_A, True), b)

    def cho_factor(cls, A):
        if cls.use_theano:
            return tt.slinalg.cholesky(A)
        else:
            return scipy.linalg.cholesky(A, lower=True)

    def RandomStreams(cls, *args, **kwargs):
        if cls.use_theano:
            return tt.shared_randomstreams.RandomStreams(*args, **kwargs)
        else:
            return _RandomStreams(*args, **kwargs)

    @property
    def pi(cls):
        return _PI

    def __getattr__(cls, attr):
        if cls.use_theano:
            return getattr(tt, attr)
        else:
            return getattr(np, attr)


class theano_math(metaclass=MathType):
    """Alias for ``numpy`` or ``theano.tensor``."""

    use_theano = True


class numpy_math(metaclass=MathType):
    """Alias for ``numpy`` or ``theano.tensor``."""

    use_theano = False
