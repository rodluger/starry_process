import numpy as np
from scipy.linalg import cho_factor


def norm_cov_sampling(mu, cov, nsamples=1000000):
    """
    Numerical estimate of normalized covariance from samples.

    """
    K = cov.shape[0]
    u = np.random.randn(K, nsamples)
    L = np.tril(cho_factor(cov, lower=True)[0])

    # Draw a bunch of samples
    x = mu + L @ u

    # Normalize each one to its mean
    xnorm = x / np.mean(x, axis=0).reshape(1, -1)

    # Compute the sample covariance
    return np.cov(xnorm)


def norm_cov_series(mu, cov, N=10):
    """
    Series approximation to the normalized covariance.

    """
    # Terms
    K = cov.shape[0]
    j = np.ones((K, 1))
    m = np.mean(cov)
    q = (cov @ j) / (K * m)
    z = m / mu ** 2

    # Coefficients
    fac = 1.0
    alpha = 0.0
    beta = 0.0
    for n in range(0, N + 1):
        alpha += fac
        beta += 2 * n * fac
        fac *= z * (2 * n + 3)

    # We're done
    return (alpha / mu ** 2) * cov + z * (
        (alpha + beta) * (j - q) @ (j - q).T - alpha * q @ q.T
    )


def test_norm_cov():
    """
    Compare our expression for the covariance of a normalized
    Gaussian process to an estimate from direct sampling.

    """
    np.random.seed(0)

    # GP mean
    mu = 0.75

    # Dimension of the problem
    K = 3

    # Random covariance and its Cholesky decomp.
    L = 0.1 * np.tril(0.25 * np.random.randn(K, K) + np.eye(K))
    cov = L @ L.T

    # Compare the two estimates
    cov1 = norm_cov_series(mu, cov)
    cov2 = norm_cov_sampling(mu, cov)
    assert np.allclose(cov1, cov2, atol=1e-4)
