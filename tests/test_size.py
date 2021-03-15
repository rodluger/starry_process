from starry_process.size import SizeIntegral
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm
from theano.configparser import change_flags
from starry_process.compat import theano, tt


def test_size(ydeg=15, r=15.0, dr=5.0, rtol=1e-7, ftol=1e-7, **kwargs):
    # Indices of non-zero elements
    l = np.arange(ydeg + 1)
    i = l * (l + 1)
    ij = np.ix_(i, i)

    # Get analytic integrals
    print("Computing moments analytically...")
    S = SizeIntegral(r, dr, ydeg=ydeg, **kwargs)
    e = S._first_moment().eval()[i]
    eigE = S._second_moment().eval()
    E = (eigE @ eigE.T)[ij]

    # Get the first moment by numerical integration
    r_rad = r * np.pi / 180
    d_rad = dr * np.pi / 180
    e_num = np.zeros(ydeg + 1)
    Bp = S._spot.Bp.eval()
    theta = S._spot.theta.eval()
    sfac = S._spot.sfac
    b = lambda rho: 1 / (1 + np.exp(sfac * (rho - theta))) - 1
    print("Computing first moment numerically...")
    for l in tqdm(range(ydeg + 1)):

        def func(rho):
            return np.inner(Bp[l], b(rho))

        e_num[l] = quad(func, r_rad - d_rad, r_rad + d_rad)[0] / (2 * d_rad)

    # Get the second moment by numerical integration
    E_num = np.zeros((ydeg + 1, ydeg + 1))
    print("Computing second moment numerically...")
    for l1 in tqdm(range(ydeg + 1)):
        for l2 in range(ydeg + 1):

            def func(rho):
                s1 = np.inner(Bp[l1], b(rho))
                s2 = np.inner(Bp[l2], b(rho))
                return s1 * s2

            E_num[l1, l2] = quad(func, r_rad - d_rad, r_rad + d_rad)[0] / (
                2 * d_rad
            )

    # Compare
    assert np.max(np.abs(e - e_num)) < rtol, "error in first moment"
    assert np.max(np.abs(1 - e / e_num)) < ftol, "error in first moment"
    assert np.max(np.abs(E - E_num)) < rtol, "error in second moment"
    assert np.max(np.abs(1 - E / E_num)) < ftol, "error in second moment"


def test_size_grad(
    ydeg=15, r=15.0, dr=5.0, abs_tol=1e-5, rel_tol=1e-5, eps=1e-5
):
    with change_flags(compute_test_value="off"):

        S = SizeIntegral(r, dr, ydeg=ydeg)

        # d/de
        theano.gradient.verify_grad(
            lambda r, dr: SizeIntegral(r, dr, ydeg=ydeg)._first_moment(),
            (r, dr),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            rng=np.random,
        )

        # d/dE
        theano.gradient.verify_grad(
            lambda r, dr: SizeIntegral(r, dr, ydeg=ydeg)._second_moment(),
            (r, dr),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
            rng=np.random,
        )
