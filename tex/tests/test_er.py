import numpy as np
from scipy.integrate import quad
from scipy.special import legendre as P


def b(r, K=1000, s=0.0033 * 180 / np.pi, **kwargs):
    """
    The sigmoid spot profile.

    """
    theta = np.linspace(0, np.pi, K)
    return 1 / (1 + np.exp((r - theta) / s)) - 1


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


def er_dzero(r, s=0.0033 * 180 / np.pi, **kwargs):
    """Return the longitude expectation integral e_r for delta r = 0."""
    return get_Bp() @ b(r)


def er(r, dr, s=0.0033 * 180 / np.pi, **kwargs):
    """Return the longitude expectation integral e_r for delta r > 0."""
    theta = np.linspace(0, np.pi, kwargs.get("K", 1000))
    num = 1 + np.exp((r - dr - theta) / s)
    den = 1 + np.exp((r + dr - theta) / s)
    return (s / (2 * dr)) * get_Bp(**kwargs) @ np.log(num / den)


def er_numerical(r, dr, **kwargs):
    """Return the longitude expectation integral e_r, computed numerically."""
    lmax = kwargs.get("lmax", 5)
    Bp = get_Bp(**kwargs)
    stilde = lambda r, l: np.inner(Bp[l], b(r, **kwargs))
    return [
        (1.0 / (2 * dr)) * quad(stilde, r - dr, r + dr, args=l)[0]
        for l in range(lmax + 1)
    ]


def test_er():
    """
    Show that our expression for the first moment
    integral of the radius distribution agrees
    with a numerical estimate.
    """
    # Check that our analytic expression agrees with the
    # numerical integral
    r = 20 * np.pi / 180
    dr = 5 * np.pi / 180
    assert np.allclose(er(r, dr), er_numerical(r, dr))

    # Check our expression in the limit dr --> 0
    r = 20 * np.pi / 180
    dr = 1e-8
    assert np.allclose(er_dzero(r), er_numerical(r, dr))
