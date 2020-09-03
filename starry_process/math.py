import theano
import theano.tensor as tt
from theano.tensor import slinalg, nlinalg
from theano.ifelse import ifelse
import numpy as np
from inspect import getmro


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


def cast(*args):
    if len(args) == 1:
        return tt.as_tensor_variable(args[0]).astype(tt.config.floatX)
    else:
        return [
            tt.as_tensor_variable(arg).astype(tt.config.floatX) for arg in args
        ]


def eigen(Q, neig=None, driver=None):
    """
    Returns the matrix square root of `Q`,
    computed via (hermitian) eigendecomposition:

        eigen(Q) . eigen(Q)^T = Q

    """
    w, U = nlinalg.eigh(Q)
    U = tt.dot(U, tt.diag(tt.sqrt(tt.maximum(0, w))))
    if neig is not None:
        return U[:, -neig:]
    else:
        return U

    # TODO: implement the scipy version
    """
    N = Q.shape[0]
    if neig is None:
        neig = N
    w, U = scipy.linalg.eigh(Q, subset_by_index=(N - neig, N - 1))
    U = U @ np.diag(np.sqrt(np.maximum(0, w)))
    return U[:, ::-1]
    """
