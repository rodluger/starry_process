import numpy as np
from scipy.integrate import quad
from scipy.special import legendre as P


def b(r, K=1000, s=0.0033, **kwargs):
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


def bigEr_dzero(r, s=0.0033, **kwargs):
    """Return the longitude expectation integral E_r for delta r = 0."""
    L = (get_Bp(**kwargs) @ b(r)).reshape(-1, 1)
    return L @ L.T


def bigEr(r, dr, s=0.0033, cutoff=1.5, **kwargs):
    """Return the longitude expectation integral E_r for delta r > 0."""
    # Generate an array in `theta`
    # NOTE: For theta > r + dr, the `C` matrix drops
    # to zero VERY quickly. In practice we get better
    # numerical stability if we just set those elements
    # to zero without evaluating them, especially since
    # we can get NaNs from operations involving the extremely
    # large dynamic range between the `exp` and `ln` terms.
    # TODO: It's likely we can find a more numerically stable
    # expression for `C`...
    K = kwargs.get("K", 1000)
    theta = np.linspace(0, np.pi, K).reshape(-1, 1)
    kmax = np.argmax(theta / (r + dr) > cutoff)

    chim = np.exp((r - dr - theta[:kmax]) / s)
    chip = np.exp((r + dr - theta[:kmax]) / s)
    exp = np.exp((theta[:kmax] - theta[:kmax].T) / s)
    term = np.log((1 + chim) / (1 + chip))
    C0 = (exp * term - term.T) / (1 - exp)

    # When k = kp, we must take the limit, given below
    C0[np.diag_indices_from(C0)] = (
        1 / (1 + chip) + chim / (1 + chim) - term - 1
    ).flatten()

    # Normalization
    C0 *= s / (2 * dr)

    # Fill in the full matrix
    C = np.zeros((K, K))
    C[:kmax, :kmax] = C0

    # Rotate into ylm space
    Bp = get_Bp(**kwargs)
    return Bp @ C @ Bp.T


def bigEr_numerical(r, dr, **kwargs):
    """Return the longitude expectation integral E_r, computed numerically."""
    lmax = kwargs.get("lmax", 5)
    Bp = get_Bp(**kwargs)
    integrand = lambda r, llp: np.inner(Bp[llp[0]], b(r, **kwargs)) * np.inner(
        Bp[llp[1]], b(r, **kwargs)
    )
    return [
        [
            (1.0 / (2 * dr)) * quad(integrand, r - dr, r + dr, args=[l, lp])[0]
            for l in range(lmax + 1)
        ]
        for lp in range(lmax + 1)
    ]


def test_bigEr():
    """
    Show that our expression for the second moment
    integral of the radius distribution agrees
    with a numerical estimate.
    """
    # Check that our analytic expression agrees with the
    # numerical integral
    r = 20 * np.pi / 180
    dr = 5 * np.pi / 180
    assert np.allclose(bigEr(r, dr), bigEr_numerical(r, dr))

    # Check our expression in the limit dr --> 0
    r = 20 * np.pi / 180
    dr = 1e-8
    assert np.allclose(bigEr_dzero(r), bigEr_numerical(r, dr))
