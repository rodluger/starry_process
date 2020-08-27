from starry_process.size import SizeIntegral
from starry_process.ops.size import SizeIntegralOp
import numpy as np
from scipy.stats import beta as Beta
import theano
import theano.tensor as tt


def test_size(
    ydeg=15, alpha=10.0, beta=30.0, nsamples=int(1e7), rtol=1e-5, ftol=1e-2
):

    # Get analytic integral
    I = SizeIntegral(ydeg=ydeg)
    I._set_params(alpha, beta)
    e = I._first_moment()
    eigE = I._second_moment()
    E = tt.dot(eigE, tt.transpose(eigE))
    e = e.eval()
    E = E.eval()

    # The m != 0 terms of this integral should all be zero
    l = np.arange(ydeg + 1)
    i = l * (l + 1)
    ij = np.ix_(i, i)
    n = np.arange((ydeg + 1) ** 2)
    not_i = np.delete(n, i)
    not_ij = np.ix_(not_i, not_i)
    assert np.all(np.abs(e[not_i]) < 1e-15)
    assert np.all(np.abs(E[not_ij]) < 1e-15)

    # Retain only the m = 0 terms
    e = e[i]
    E = E[ij]

    # Integrate by sampling
    np.random.seed(0)
    rho = Beta.rvs(alpha, beta, size=nsamples)
    s = I.transform.get_s(rho=rho)[:, i]

    # Empirical moments
    e_num = np.mean(s, axis=0)
    E_num = np.cov(s.T) + np.outer(e_num, e_num)

    # Compare
    assert np.max(np.abs(e - e_num)) < rtol, "error in first moment"
    assert np.max(np.abs(1 - e / e_num)) < ftol, "error in first moment"
    assert np.max(np.abs(E - E_num)) < rtol, "error in second moment"
    assert np.max(np.abs(1 - E / E_num)) < ftol, "error in second moment"
