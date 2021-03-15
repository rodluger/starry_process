from starry_process.math import matrix_sqrt
from starry_process.ops import EighOp, LatitudeIntegralOp
import numpy as np
from theano.configparser import change_flags
from starry_process.compat import theano, tt
import pytest


def test_sqrt_grad():
    with change_flags(compute_test_value="off"):
        np.random.seed(0)
        Q = np.random.randn(10, 10)
        Q = Q @ Q.T
        theano.gradient.verify_grad(
            lambda x: tt.sum(matrix_sqrt(x)), (Q,), n_tests=1, rng=np.random
        )


def test_eigh_grad():
    with change_flags(compute_test_value="off"):
        np.random.seed(0)
        Q = np.random.randn(10, 10)
        Q = Q @ Q.T
        eigh = EighOp()
        # Test the eigenvalues
        theano.gradient.verify_grad(
            lambda x: tt.sum(eigh(x)[0]), (Q,), n_tests=1, rng=np.random
        )
        # Test the eigenvectors
        theano.gradient.verify_grad(
            lambda x: tt.sum(eigh(x)[1]), (Q,), n_tests=1, rng=np.random
        )


@pytest.mark.xfail
def test_sqrt_grad_low_rank():
    # NOTE: For ydeg > 2 the *numerical* gradient gets
    # very unstable, so I'm not sure how to test this!
    ydeg = 2

    # Let's compute the sqrt of the latitude integral Q
    alpha = 55.0
    beta = 7.5

    def Q(alpha, beta):
        return LatitudeIntegralOp(ydeg)(alpha, beta)[3]

    def U(alpha, beta):
        return matrix_sqrt(Q(alpha, beta), neig=2 * ydeg + 1)

    with change_flags(compute_test_value="off"):
        theano.gradient.verify_grad(
            U, (alpha, beta), n_tests=1, eps=1e-4, rng=np.random
        )


@pytest.mark.xfail
def test_eigh_grad_low_rank():
    """
    TODO: Is this failing test actually an issue? As long as the likelihood
    gradients are passing (`test_lnlike.py`), should we care about this?
    The eigenvectors are defined only up to a multiplicative constant, so
    perhaps the gradient is itself ill defined on its own? The likelihood
    tests show that our implementation of the matrix square root within the
    latitude and size integrals yields the correct gradients,
    so perhaps this test is unnecessary. Also, we should keep in mind that
    the numerical gradient here is *extremely* unstable.
    """
    with change_flags(compute_test_value="off"):
        np.random.seed(0)
        Q = np.random.randn(10, 3)
        Q = Q @ Q.T
        eigh = EighOp(neig=3)
        # Test the eigenvalues
        theano.gradient.verify_grad(
            lambda x: tt.sum(eigh(x)[0]), (Q,), n_tests=1
        )
        # Test the eigenvectors
        theano.gradient.verify_grad(
            lambda x: eigh(x)[1][0, 0], (Q,), n_tests=1, rng=np.random
        )
