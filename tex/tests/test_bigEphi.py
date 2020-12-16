import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import beta as B
from scipy.special import hyp2f1
from scipy.integrate import quad
from scipy.stats import beta as Beta
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


def Qllpiip(l, lp, i, ip, alpha, beta):
    if (i + ip) % 2 == 0:
        return B(alpha, l + lp + beta - (i + ip) / 2) * hyp2f1(
            -(i + ip) / 2, alpha, l + lp + alpha + beta - (i + ip) / 2, -1
        )
    else:
        return 0.0


def Pllpmmp(l, lp, m, mp, alpha, beta, Ellpr):
    term1 = 0
    for mu in range(-l, l + 1):
        for mup in range(-lp, lp + 1):
            term2 = 0
            for i in range(2 * l + 1):
                for ip in range(2 * lp + 1):
                    term2 += (
                        clmmpi(l, m, mu, i)
                        * clmmpi(lp, mp, mup, ip)
                        * Qllpiip(l, lp, i, ip, alpha, beta)
                    )
            term1 += (
                Ellpr[l + mu, lp + mup]
                * np.exp(1j * np.pi / 2 * (m - mu + mp - mup))
                * term2
            )
    return term1 / B(alpha, beta)


def Pllp(l, lp, alpha, beta, Ellpr):
    P = np.zeros((2 * l + 1, 2 * lp + 1), dtype="complex128")
    for m in range(-l, l + 1):
        for mp in range(-lp, lp + 1):
            P[l + m, lp + mp] = Pllpmmp(l, lp, m, mp, alpha, beta, Ellpr)
    return P


def Ellpphi(l, lp, alpha, beta, Ellpr):
    return (
        np.linalg.inv(Ul(l))
        @ Pllp(l, lp, alpha, beta, Ul(l) @ Ellpr @ Ul(lp).T)
        @ np.linalg.inv(Ul(lp)).T
    )


def bigEphi(alpha, beta, Er):
    lmax = int(np.sqrt(Er.shape[0]) - 1)
    N = (lmax + 1) ** 2
    E = np.zeros((N, N), dtype="complex128")
    for l in tqdm(
        range(lmax + 1), disable=bool(int(os.getenv("NOTQDM", "0")))
    ):
        for lp in range(lmax + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            j = slice(lp ** 2, (lp + 1) ** 2)
            E[i, j] = Ellpphi(l, lp, alpha, beta, Er[i, j])
    assert np.max(np.abs(E.imag)) < 1e-15
    return E.real


def bigEphi_numerical(alpha, beta, Er):
    lmax = int(np.sqrt(Er.shape[0]) - 1)
    N = (lmax + 1) ** 2
    E = np.zeros((N, N))
    for n1 in tqdm(range(N), disable=bool(int(os.getenv("NOTQDM", "0")))):
        for n2 in range(N):

            def func(phi):
                Rl = R(lmax, 0.5 * np.pi, phi, -0.5 * np.pi)
                Rs = np.zeros((N, N))
                for l in range(lmax + 1):
                    for lp in range(lmax + 1):
                        i = slice(l ** 2, (l + 1) ** 2)
                        j = slice(lp ** 2, (lp + 1) ** 2)
                        Rs[i, j] = Rl[l] @ Er[i, j] @ Rl[lp].T
                jac = 0.5 * np.abs(np.sin(phi))
                return Rs[n1, n2] * jac * Beta.pdf(np.cos(phi), alpha, beta)

            E[n1, n2] = quad(func, -np.pi / 2, np.pi / 2)[0]

    return E


def test_bigEphi(lmax=3, alpha=2.0, beta=5.0):
    np.random.seed(0)
    Er = np.random.randn((lmax + 1) ** 2, (lmax + 1) ** 2)
    Er = Er + Er.T
    Ephi1 = bigEphi(alpha, beta, Er)
    Ephi2 = bigEphi_numerical(alpha, beta, Er)
    assert np.allclose(Ephi1, Ephi2)

