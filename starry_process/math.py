import theano.tensor as tt
from theano.tensor import nlinalg as nla
from theano.tensor import slinalg as sla
from theano.ifelse import ifelse


__all__ = ["eigen", "cho_factor", "cho_solve"]


def eigen(Q, neig=None):
    """
    Returns the square root of a hermitian matrix `Q`,
    computed via eigendecomposition, such that:

        eigen(Q) . eigen(Q)^T = Q

    """
    # TODO: Is there a way to compute only `neig` eigenvalues?
    w, U = nla.eigh(Q)
    U = tt.dot(U, tt.diag(tt.sqrt(tt.maximum(0, w))))
    if neig is not None:
        return U[:, -neig:]
    else:
        return U


# Cholesky solve
_solve_lower = sla.Solve(A_structure="lower_triangular", lower=True)
_solve_upper = sla.Solve(A_structure="upper_triangular", lower=False)


def cho_solve(cho_A, b):
    return _solve_upper(tt.transpose(cho_A), _solve_lower(cho_A, b))


def cho_factor(A):
    return sla.cholesky(A)
