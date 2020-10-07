from starry_process.size import SizeIntegral
from starry_process.ops.size import SizeIntegralOp
import numpy as np
from scipy.stats import beta as Beta
from scipy.integrate import quad
from tqdm import tqdm
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags


def test_size(ydeg=15, params=[0.5, 0.5], rtol=1e-10, ftol=1e-7, **kwargs):

    # Get analytic integral
    print("Computing moments analytically...")
    I = SizeIntegral(params, ydeg=ydeg, **kwargs)
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
    alpha, beta = I._transform._ab_to_alphabeta(*params)
    e_num = np.zeros(ydeg + 1)
    print("Computing first moment numerically...")
    for l in tqdm(range(ydeg + 1)):

        n = l * (l + 1)

        def func(rho):
            s = I._transform.get_s(rho=rho)[0]
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
                s = I._transform.get_s(rho=rho)[0]
                return s[n1] * s[n2] * Beta.pdf(rho, alpha, beta)

            E_num[l1, l2] = quad(func, 0, 1, epsabs=1e-12, epsrel=1e-12)[0]

    # Compare
    assert np.max(np.abs(e - e_num)) < rtol, "error in first moment"
    assert np.max(np.abs(1 - e / e_num)) < ftol, "error in first moment"
    assert np.max(np.abs(E - E_num)) < rtol, "error in second moment"
    assert np.max(np.abs(1 - E / E_num)) < ftol, "error in second moment"


def test_size_grad(
    ydeg=15, params=[0.5, 0.5], abs_tol=1e-5, rel_tol=1e-5, eps=1e-7,
):
    with change_flags(compute_test_value="off"):
        op = SizeIntegralOp(ydeg)

        # Get Beta params
        a, b = params
        a1 = -5
        a2 = 5
        b1 = -5
        b2 = 5
        alpha = np.exp(a * (a2 - a1) + a1)
        beta = np.exp(b * (b2 - b1) + b1)

        # d/dq
        verify_grad(
            lambda alpha, beta: op(alpha, beta)[0],
            (alpha, beta,),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
        )

        # d/dQ
        verify_grad(
            lambda alpha, beta: op(alpha, beta)[3],
            (alpha, beta,),
            n_tests=1,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            eps=eps,
        )


if __name__ == "__main__":
    test_size_grad()
