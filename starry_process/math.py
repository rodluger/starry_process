import theano.tensor as tt
from theano.tensor import slinalg, nlinalg

# Force double precision in Theano
tt.config.floatX = "float64"
tt.config.cast_policy = "numpy+floatX"


# Theano Cholesky solve
_solve_lower = slinalg.Solve(A_structure="lower_triangular", lower=True)
_solve_upper = slinalg.Solve(A_structure="upper_triangular", lower=False)


def cho_solve(cho_A, b):
    return _solve_upper(tt.transpose(cho_A), _solve_lower(cho_A, b))


def cho_factor(A):
    return slinalg.cholesky(A)


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
