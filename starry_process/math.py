from .ops import EighOp
import theano
import theano.tensor as tt
from theano.tensor import slinalg
from theano.ifelse import ifelse
import numpy as np
from inspect import getmro


__all__ = ["is_tensor", "cho_solve", "cho_factor", "cast", "matrix_sqrt"]


# Force double precision in Theano
tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"


def is_tensor(*objs):
    """Return ``True`` if any of ``objs`` is a ``Theano`` object."""
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def cho_solve(cho_A, b):
    solve_lower = slinalg.Solve(A_structure="lower_triangular", lower=True)
    solve_upper = slinalg.Solve(A_structure="upper_triangular", lower=False)
    return ifelse(
        tt.any(tt.isnan(cho_A)),
        tt.ones_like(b),
        solve_upper(tt.transpose(cho_A), solve_lower(cho_A, b)),
    )


def cho_factor(A):
    cholesky = slinalg.Cholesky(on_error="nan")
    return ifelse(tt.any(tt.isnan(A)), tt.ones_like(A) * np.nan, cholesky(A))


def cast(*args, vectorize=False):
    if vectorize:
        if len(args) == 1:
            return tt.reshape(
                tt.as_tensor_variable(args[0]).astype(tt.config.floatX), (-1,)
            )
        else:
            return [
                tt.reshape(
                    tt.as_tensor_variable(arg).astype(tt.config.floatX), (-1,)
                )
                for arg in args
            ]
    else:
        if len(args) == 1:
            return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
        else:
            return [
                tt.as_tensor_variable(arg).astype(tt.config.floatX)
                for arg in args
            ]


def matrix_sqrt(Q, neig=None, driver="numpy", mindiff=1e-15):
    """
    Returns the matrix square root of `Q`,
    computed via (hermitian) eigendecomposition:

        matrix_sqrt(Q) . matrix_sqrt(Q)^T = Q

    """
    # Eigendecomposition: eigenvalues `w` and eigenvectors `U`
    eigh = EighOp(neig=neig, driver=driver, mindiff=mindiff)
    w, U = eigh(Q)

    # Get the square root of the positive eigenvalues
    sqrtw = tt.switch(
        tt.gt(w, mindiff * tt.ones_like(w)), tt.sqrt(w), tt.zeros_like(w)
    )

    # Dot them in: the result is the matrix square root of `Q`
    return tt.dot(U, tt.diag(sqrtw))
