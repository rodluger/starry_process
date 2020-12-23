import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import hyp2f1
from scipy.integrate import quad
from wigner import R
from starry_process.ops import rTA1Op


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
    """Return the scalar (q_I^{l})_{i}."""
    return (
        (-1) ** i
        / (l - 0.5 * i + 1)
        * hyp2f1(1, -0.5 * i, 2 + l - 0.5 * i, -1)
    )


def plm(l, m, elyp):
    """Return the scalar (P_I^{l})_{m}."""
    term1 = 0
    for mu in range(-l, l + 1):
        term2 = 0
        for i in range(2 * l + 1):
            term2 += clmmpi(l, m, mu, i) * qli(l, i)
        term1 += elyp[l + mu] * np.exp(1j * np.pi / 2 * (m - mu)) * term2
    return term1


def pl(l, elyp):
    """Return the vector p_I^{l}."""
    p = np.zeros(2 * l + 1, dtype="complex128")
    for m in range(-l, l + 1):
        p[l + m] = plm(l, m, elyp)
    return p


def elypp(l, elyp):
    """Return the vector e_{y''}^{l}."""
    U = Ul(l)
    return np.linalg.inv(U) @ pl(l, U @ elyp)


def eI(ey):
    """Return the first moment integral of the inclination distribution."""
    lmax = int(np.sqrt(len(ey)) - 1)
    rTA1 = rTA1Op(ydeg=lmax)().eval()
    e = []
    Rx = R(lmax, 0.5 * np.pi, 0.5 * np.pi, -0.5 * np.pi)
    for l in range(lmax + 1):
        e.extend(elypp(l, Rx[l] @ ey[l ** 2 : (l + 1) ** 2],))
    e = np.array(e)
    assert np.max(np.abs(e.imag)) < 1e-15
    return np.inner(rTA1, e.real)


def eI_numerical(ey):
    """Return the first moment integral of the inclination distribution, computed numerically."""
    lmax = int(np.sqrt(len(ey)) - 1)
    N = (lmax + 1) ** 2
    rTA1 = rTA1Op(ydeg=lmax)().eval()
    e = np.zeros(N)

    Rx = R(lmax, 0.5 * np.pi, 0.5 * np.pi, -0.5 * np.pi)

    for n in range(N):

        def func(I):
            RxI = R(lmax, 0.5 * np.pi, -I, -0.5 * np.pi)
            Rs = np.zeros(N)
            for l in range(lmax + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                Rs[i] = RxI[l] @ Rx[l] @ ey[i]
            return Rs[n] * np.sin(I)

        e[n] = quad(func, 0, 0.5 * np.pi)[0]
    return np.inner(rTA1, e)


def test_eI(lmax=5):
    """
    Show that our expression for the first moment
    expectation integral for the inclination agrees
    with a numerical estimate. For simplicity, we're
    making use of the isotropy assumption, so this
    moment integral is actually just a scalar.
    """
    np.random.seed(0)
    ey = np.random.randn((lmax + 1) ** 2)
    assert np.allclose(eI(ey), eI_numerical(ey))
