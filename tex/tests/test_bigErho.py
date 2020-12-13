import numpy as np
from scipy.integrate import quad
from scipy.special import legendre as P


def b(rho, K=1000, s=0.0033 * 180 / np.pi, **kwargs):
    """
    The sigmoid spot profile.

    """
    theta = np.linspace(0, np.pi, K)
    return 1 / (1 + np.exp((rho - theta) / s)) - 1


def get_Bp(K=1000, lmax=5, eps=1e-9, sigma=15, **kwargs):
    """
    Return the matrix B+. This expands the 
    spot profile `b` in Legendre polynomials.

    """
    theta = np.linspace(0, np.pi, K)
    cost = np.cos(theta)
    B = np.hstack(
        [
            np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1)
            for l in range(lmax + 1)
        ]
    )
    BInv = np.linalg.solve(B.T @ B + eps * np.eye(lmax + 1), B.T)
    l = np.arange(lmax + 1)
    i = l * (l + 1)
    S = np.exp(-0.5 * i / sigma ** 2)
    BInv = S[:, None] * BInv
    return BInv


def bigErho_dzero(r, s=0.0033 * 180 / np.pi, **kwargs):
    L = (get_Bp(**kwargs) @ b(r)).reshape(-1, 1)
    return L @ L.T


def bigErho(r, d, s=0.0033 * 180 / np.pi, **kwargs):
    theta = np.linspace(0, np.pi, kwargs.get("K", 1000)).reshape(-1, 1)
    chim = np.exp((r - d - theta) / s)
    chip = np.exp((r + d - theta) / s)
    exp = np.exp((theta - theta.T) / s)
    C = (
        exp * np.log((1 + chim) / (1 + chip))
        - np.log((1 + chim) / (1 + chip)).T
    ) / (1 - exp)

    # When k = kp, we must take the limit, given below
    C[np.diag_indices_from(C)] = (
        1 / (1 + chip)
        + chim / (1 + chim)
        - np.log((1 + chim) / (1 + chip))
        - 1
    ).flatten()

    # Normalization
    C *= s / (2 * d)

    Bp = get_Bp(**kwargs)
    return Bp @ C @ Bp.T


def bigErho_numerical(r, d, **kwargs):
    lmax = kwargs.get("lmax", 5)
    Bp = get_Bp(**kwargs)
    integrand = lambda rho, llp: np.inner(
        Bp[llp[0]], b(rho, **kwargs)
    ) * np.inner(Bp[llp[1]], b(rho, **kwargs))
    return [
        [
            (1.0 / (2 * d)) * quad(integrand, r - d, r + d, args=[l, lp])[0]
            for l in range(lmax + 1)
        ]
        for lp in range(lmax + 1)
    ]


def test_bigErho():
    # Check that our analytic expression agrees with the
    # numerical integral
    r = 20 * np.pi / 180
    d = 5 * np.pi / 180
    assert np.allclose(bigErho(r, d), bigErho_numerical(r, d))

    # Check our expression in the limit d --> 0
    r = 20 * np.pi / 180
    d = 1e-8
    assert np.allclose(bigErho_dzero(r), bigErho_numerical(r, d))
