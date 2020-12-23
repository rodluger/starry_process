import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import beta as B
from scipy.special import hyp2f1
from scipy.integrate import quad
from scipy.stats import beta as Beta
from wigner import R


def factorial(n):
    return Gamma(n + 1)


def Ul(l):
    """
    Return the transformation matrix from complex to real Ylms.
    
    """
    U = np.zeros((2 * l + 1, 2 * l + 1), dtype="complex128")
    for m in range(-l, l + 1):
        for mu in range(-l, l + 1):
            if mu < 0:
                term1 = 1j
            elif mu == 0:
                term1 = np.sqrt(2) / 2
            else:
                term1 = 1
            if (m > 0) and (mu < 0) and (mu % 2 == 0):
                term2 = -1
            elif (m > 0) and (mu > 0) and (mu % 2 != 0):
                term2 = -1
            else:
                term2 = 1
            U[l + m, l + mu] = (
                term1 * term2 * 1 / np.sqrt(2) * (int(m == mu) + int(m == -mu))
            )
    return U


def clmmpi(l, m, mp, i):
    """Return the scalar c^l_{m,mu,i}."""
    if (m - mp - i) % 2 == 0:
        return (
            (-1) ** ((2 * l - m + mp - i) / 2)
            * np.sqrt(
                factorial(l - m)
                * factorial(l + m)
                * factorial(l - mp)
                * factorial(l + mp)
            )
            / (
                2 ** l
                * factorial((i - m - mp) / 2)
                * factorial((i + m + mp) / 2)
                * factorial((2 * l - i - m + mp) / 2)
                * factorial((2 * l - i + m - mp) / 2)
            )
        )
    else:
        return 0.0


def qli(l, i):
    """Return the scalar (q_\lambda^{l})_{i}."""
    if i % 2 == 0:
        return (
            2 ** l
            / np.pi
            * Gamma(0.5 * (1 + i))
            * Gamma(l + 0.5 * (1 - i))
            / Gamma(l + 1)
        )
    else:
        return 0.0


def plm(l, m, elphi):
    """Return the scalar (p_\lambda^{l})_{m}."""
    term1 = 0
    for mu in range(-l, l + 1):
        term2 = 0
        for i in range(2 * l + 1):
            term2 += clmmpi(l, m, mu, i) * qli(l, i)
        term1 += elphi[l + mu] * term2
    return term1


def pl(l, elphi):
    """Return the vector p_\lambda^{l}."""
    p = np.zeros(2 * l + 1, dtype="complex128")
    for m in range(-l, l + 1):
        p[l + m] = plm(l, m, elphi)
    return p


def ellam(l, elphi):
    """Return the vector e_\lambda^{l}."""
    U = Ul(l)
    return np.linalg.inv(U) @ pl(l, U @ elphi)


def elam(ephi):
    """Return the first moment integral of the longitude."""
    lmax = int(np.sqrt(len(ephi)) - 1)
    e = []
    for l in range(lmax + 1):
        e.extend(ellam(l, ephi[l ** 2 : (l + 1) ** 2]))
    e = np.array(e)
    assert np.max(np.abs(e.imag)) < 1e-15
    return e.real


def elam_numerical(ephi):
    """Return the first moment integral of the longitude, computed numerically."""
    lmax = int(np.sqrt(len(ephi)) - 1)
    N = (lmax + 1) ** 2
    e = np.zeros(N)

    for n in range(N):

        def func(lam):
            Rl = R(lmax, 0, lam, 0)
            Rs = np.zeros(N)
            for l in range(lmax + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                Rs[i] = Rl[l] @ ephi[i]
            return Rs[n] / (2 * np.pi)

        e[n] = quad(func, -np.pi, np.pi)[0]

    return e


def test_elam(lmax=5):
    """
    Show that our expression for the first moment
    integral of the longitude distribution agrees
    with a numerical estimate.
    """
    np.random.seed(0)
    ephi = np.random.randn((lmax + 1) ** 2)
    assert np.allclose(elam(ephi), elam_numerical(ephi))
