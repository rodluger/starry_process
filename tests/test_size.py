from starry_process.size import SizeIntegral
from starry_process.ops.size import SizeIntegralOp
import numpy as np
from scipy.stats import beta as Beta
from scipy.integrate import quad
from tqdm import tqdm


# TODO: alpha = 10, beta = 20 is unstable for some reason!
# Use kwarg `SP_COMPUTE_G_NUMERICALLY=1` to circumvent this


def test_size(ydeg=15, alpha=1.0, beta=50.0, rtol=1e-10, ftol=1e-7, **kwargs):

    # Get analytic integral
    print("Computing moments analytically...")
    I = SizeIntegral(ydeg=ydeg, **kwargs)
    I._set_params(alpha, beta)
    e = I._first_moment().eval()
    eigE = I._second_moment().eval()
    E = eigE @ eigE.T

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

    # Get the first moment by numerical integration
    e_num = np.zeros(ydeg + 1)
    print("Computing first moment numerically...")
    for l in tqdm(range(ydeg + 1)):

        n = l * (l + 1)

        def func(rho):
            s = I.transform.get_s(rho=rho)[0]
            return s[n] * Beta.pdf(rho, alpha, beta)

        e_num[l] = quad(func, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]

    # Get the second moment by numerical integration
    E_num = np.zeros((ydeg + 1, ydeg + 1))
    print("Computing second moment numerically...")
    for l1 in tqdm(range(ydeg + 1)):

        n1 = l1 * (l1 + 1)

        for l2 in range(ydeg + 1):

            n2 = l2 * (l2 + 1)

            def func(rho):
                s = I.transform.get_s(rho=rho)[0]
                return s[n1] * s[n2] * Beta.pdf(rho, alpha, beta)

            E_num[l1, l2] = quad(func, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]

    # Compare
    assert np.max(np.abs(e - e_num)) < rtol, "error in first moment"
    assert np.max(np.abs(1 - e / e_num)) < ftol, "error in first moment"
    assert np.max(np.abs(E - E_num)) < rtol, "error in second moment"
    assert np.max(np.abs(1 - E / E_num)) < ftol, "error in second moment"


def test_python(ydeg=15, alpha=1.0, beta=50.0, **kwargs):
    # Get analytic integrals (theano)
    print("Computing moments using theano...")
    I = SizeIntegral(ydeg=ydeg, **kwargs)
    I._set_params(alpha, beta)
    e = I._first_moment().eval()
    eigE = I._second_moment().eval()
    E = eigE @ eigE.T

    # Get analytic integrals (python)
    print("Computing moments using python...")
    I = SizeIntegral(ydeg=ydeg, use_theano=False, **kwargs)
    I._set_params(alpha, beta)
    e_python = I._first_moment()
    eigE = I._second_moment()
    E_python = eigE @ eigE.T

    # Compare
    assert np.allclose(e, e_python), "error in first moment"
    assert np.allclose(E, E_python), "error in second moment"
