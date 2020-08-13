from starry_gp.contrast import ContrastIntegral
import numpy as np
from scipy.stats import lognorm as LogNormal


def test_contrast(ydeg=5, mu=-0.1, nu=0.01):

    # Settings
    np.random.seed(0)
    nsamples = int(1e8)
    atol = 2.0e-5

    # Random input moments
    np.random.seed(0)
    N = (ydeg + 1) ** 2
    s = np.random.randn(N)
    eigS = np.random.randn(N, N) / N
    S = eigS @ eigS.T

    # Get analytic integral
    I = ContrastIntegral(ydeg)
    I.set_params(mu, nu)
    e = I.first_moment(s)
    eigE = I.second_moment(eigS)
    E = eigE @ eigE.T

    # Integrate numerically
    y = np.zeros((N, nsamples))

    # Draw the spot contrast
    b = LogNormal.rvs(scale=np.exp(mu), s=np.sqrt(nu), size=nsamples)
    xi = 1 - b

    # Compute the moments
    e_num = np.mean(xi) * s
    E_num = np.mean(xi ** 2) * S

    # Compare
    assert np.allclose(e, e_num, atol=atol), "error in first moment"
    assert np.allclose(E, E_num, atol=atol), "error in second moment"
