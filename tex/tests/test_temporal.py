from scipy.linalg import block_diag
import numpy as np


def test_temporal():
    """
    Show that the kronecker product of a spherical harmonic
    covariance and a temporal covariance projected into
    flux space is just the elementwise product of the
    spherical harmonic covariance projected into flux
    space and the temporal covariance.

    """
    np.random.seed(0)

    # Dimensions
    ydeg = 5
    N = (ydeg + 1) ** 2
    K = 10

    # Random flux design matrix
    A = np.random.randn(K, N)

    # Random Ylm covariance
    Ly = np.tril(np.random.randn(N, N))
    Sy = Ly @ Ly.T

    # Random temporal covariance
    Lt = np.tril(np.random.randn(K, K))
    St = Lt @ Lt.T

    # Two ways of computing the same thing
    cov1 = (A @ Sy @ A.T) * St
    cov2 = block_diag(*A) @ np.kron(St, Sy) @ block_diag(*A).T

    assert np.allclose(cov1, cov2)
