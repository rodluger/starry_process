import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import hyp2f1
from scipy.integrate import quad
from wigner import R
from tqdm.auto import tqdm
import os


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


def Qllpiip(l, lp, i, ip):
    if (i + ip) % 2 == 0:
        return (
            2 ** (l + lp)
            / np.pi
            * Gamma(0.5 * (1 + i + ip))
            * Gamma(l + lp + 0.5 * (1 - i - ip))
            / Gamma(l + lp + 1)
        )
    else:
        return 0.0


def Pllpmmp(l, lp, m, mp, Ellpphi):
    term1 = 0
    for mu in range(-l, l + 1):
        for mup in range(-lp, lp + 1):
            term2 = 0
            for i in range(2 * l + 1):
                for ip in range(2 * lp + 1):
                    term2 += (
                        clmmpi(l, m, mu, i)
                        * clmmpi(lp, mp, mup, ip)
                        * Qllpiip(l, lp, i, ip)
                    )
            term1 += Ellpphi[l + mu, lp + mup] * term2
    return term1


def Pllp(l, lp, Ellpphi):
    P = np.zeros((2 * l + 1, 2 * lp + 1), dtype="complex128")
    for m in range(-l, l + 1):
        for mp in range(-lp, lp + 1):
            P[l + m, lp + mp] = Pllpmmp(l, lp, m, mp, Ellpphi)
    return P


def Ellplam(l, lp, Ellpphi):
    return (
        np.linalg.inv(Ul(l))
        @ Pllp(l, lp, Ul(l) @ Ellpphi @ Ul(lp).T)
        @ np.linalg.inv(Ul(lp)).T
    )


def bigElam(Ephi):
    lmax = int(np.sqrt(Ephi.shape[0]) - 1)
    N = (lmax + 1) ** 2
    E = np.zeros((N, N), dtype="complex128")
    for l in tqdm(
        range(lmax + 1), disable=bool(int(os.getenv("NOTQDM", "0")))
    ):
        for lp in range(lmax + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            j = slice(lp ** 2, (lp + 1) ** 2)
            E[i, j] = Ellplam(l, lp, Ephi[i, j])
    assert np.max(np.abs(E.imag)) < 1e-15
    return E.real


def bigElam_numerical(Ephi):
    lmax = int(np.sqrt(Ephi.shape[0]) - 1)
    N = (lmax + 1) ** 2
    E = np.zeros((N, N))
    for n1 in tqdm(range(N), disable=bool(int(os.getenv("NOTQDM", "0")))):
        for n2 in range(N):

            def func(lam):
                Rl = R(lmax, 0, lam, 0)
                Rs = np.zeros((N, N))
                for l in range(lmax + 1):
                    for lp in range(lmax + 1):
                        i = slice(l ** 2, (l + 1) ** 2)
                        j = slice(lp ** 2, (lp + 1) ** 2)
                        Rs[i, j] = Rl[l] @ Ephi[i, j] @ Rl[lp].T
                return Rs[n1, n2] / (2 * np.pi)

            E[n1, n2] = quad(func, -np.pi, np.pi)[0]

    return E


def test_bigElam(lmax=2):
    np.random.seed(0)
    Ephi = np.random.randn((lmax + 1) ** 2, (lmax + 1) ** 2)
    Ephi = Ephi + Ephi.T

    Elam1 = bigElam(Ephi)
    Elam2 = bigElam_numerical(Ephi)

    assert np.allclose(Elam1, Elam2)

