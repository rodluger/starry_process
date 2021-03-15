from starry_process.ops import AlphaBetaOp
from starry_process import StarryProcess
import numpy as np
from theano.configparser import change_flags
from starry_process.compat import theano, tt


def test_norm_grad():
    with change_flags(compute_test_value="off"):
        z = 0.001
        op = AlphaBetaOp(20)
        get_alpha = lambda z: op(z)[0]
        get_beta = lambda z: op(z)[1]
        theano.gradient.verify_grad(get_alpha, [z], n_tests=1, rng=np.random)
        theano.gradient.verify_grad(get_beta, [z], n_tests=1, rng=np.random)


def test_norm(ftol=0.05):
    # GP mean
    mu = 0.75

    # Dimension of the problem
    K = 3

    # Number of samples in numerical estimate
    M = 100000

    # Random covariance matrix
    np.random.seed(0)
    L = 0.1 * np.tril(0.25 * np.random.randn(K, K) + np.eye(K))
    cov = L @ L.T

    # Compute the series approximation to the normalized covariance
    cov_norm = StarryProcess()._normalize(mu, cov).eval()

    # Compute it by sampling
    u = np.random.randn(K, M)
    x = mu + L @ u
    xnorm = x / np.mean(x, axis=0).reshape(1, -1)
    cov_norm_num = np.cov(xnorm)

    # Fractional error
    error = np.abs((cov_norm - cov_norm_num) / cov_norm)
    try:
        assert np.all(error < ftol)
    except AssertionError as e:
        print(cov_norm)
        print(cov_norm_num)
        raise e
