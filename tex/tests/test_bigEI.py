import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import hyp2f1
from scipy.integrate import quad
from wigner import R
from starry_process.ops import rTA1Op
import starry
from tqdm.auto import tqdm
import os


def factorial(n):
    return Gamma(n + 1)


def Ul(l):
    """Return the transformation matrix from complex to real Ylms."""
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


def Qllpiip(l, lp, i, ip):
    """Return the scalar (Q_I^{l,l'})_{i,i'}."""
    return (
        (-1) ** (i + ip)
        / (l + lp - 0.5 * (i + ip) + 1)
        * hyp2f1(1, -0.5 * (i + ip), 2 + l + lp - 0.5 * (i + ip), -1)
    )


def Pllpkkpmmp(l, lp, k, kp, m, mp, barEyp_llpkkp):
    """Return the scalar [(P_I^{l,l'})_{k,k'}]_{m,m'}."""
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
            term1 += (
                barEyp_llpkkp[l + mu, lp + mup]
                * np.exp(1j * np.pi / 2 * (m - mu + mp - mup))
                * term2
            )
    return term1


def Pllpkkp(l, lp, k, kp, Eyp_llpkkp):
    """Return the matrix (P_I^{l,l'})_{k,k'}."""
    barEyp_llpkkp = Ul(l) @ Eyp_llpkkp @ Ul(lp).T
    P = np.zeros((2 * l + 1, 2 * lp + 1), dtype="complex128")
    for m in range(-l, l + 1):
        for mp in range(-lp, lp + 1):
            P[l + m, lp + mp] = Pllpkkpmmp(l, lp, k, kp, m, mp, barEyp_llpkkp)
    return P


def Eypp_llpkkp(l, lp, k, kp, Eyp_llpkkp):
    """Return the matrix (E_{y''}^{l,l'})_{k,k'}."""
    return (
        np.linalg.inv(Ul(l))
        @ Pllpkkp(l, lp, k, kp, Eyp_llpkkp)
        @ np.linalg.inv(Ul(lp)).T
    )


def bigEI(Ey, t, P=1):
    """Return the column kp=0 of the matrix E_I."""
    lmax = int(np.sqrt(Ey.shape[0]) - 1)
    K = len(t)
    rTA1 = rTA1Op(ydeg=lmax)().eval()
    Rx = R(lmax, 0.5 * np.pi, 0.5 * np.pi, -0.5 * np.pi)
    bigEI = np.zeros(K, dtype="complex128")
    kp = 0
    for k in tqdm(range(K), disable=bool(int(os.getenv("NOTQDM", "0")))):
        Rzk = R(lmax, 2 * np.pi / P * t[k], 0, 0)
        Rzkp = R(lmax, 2 * np.pi / P * t[kp], 0, 0)
        for l in range(lmax + 1):
            for lp in range(lmax + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                j = slice(lp ** 2, (lp + 1) ** 2)
                Eyp_llpkkp = (Rzk[l] @ Rx[l] @ Ey[i, j] @ Rx[lp].T) @ Rzkp[
                    lp
                ].T
                bigEI[k] += (
                    rTA1[i] @ Eypp_llpkkp(l, lp, k, kp, Eyp_llpkkp) @ rTA1[j].T
                )
    assert np.max(np.abs(bigEI.imag)) < 1e-15
    return bigEI.real


def bigEI_numerical(Ey, t, P=1):
    """Return the column kp=0 of the matrix E_I, computed numerically."""
    lmax = int(np.sqrt(Ey.shape[0]) - 1)
    K = len(t)
    map = starry.Map(ydeg=lmax, lazy=False)
    theta = 360 / P * t

    bigEI = np.zeros(K)
    kp = 0
    for k in tqdm(range(K), disable=bool(int(os.getenv("NOTQDM", "0")))):

        def integrand(I):
            map.inc = I * 180 / np.pi
            A = map.design_matrix(theta=theta)
            return (A @ Ey @ A.T * np.sin(I))[k, kp]

        bigEI[k] = quad(integrand, 0, 0.5 * np.pi)[0]

    return bigEI


def test_bigEI(K=50):
    """
    Show that our expression for the second moment
    expectation integral for the inclination agrees
    with a numerical estimate. For simplicity, we're
    only computing the first column of the matrix,
    which fully specifies it in the case of evenly-gridded
    data (i.e., it's Toeplitz).

    """
    t = np.linspace(0, 1, K)

    # This is the expectation integral
    # `Ey` for `ydeg = 2` and the default
    # StarryProcess settings.
    lmax = 2
    Ey = np.array(
        [
            [
                1.846895616383e-04,
                7.832444914170e-22,
                -1.776873413902e-21,
                -9.100340154061e-22,
                4.536462608839e-36,
                -4.056738113699e-37,
                2.822847908886e-05,
                -9.181931846658e-25,
                4.889316000230e-05,
            ],
            [
                7.832444914170e-22,
                4.090872345204e-04,
                1.089732939299e-19,
                1.523011171239e-19,
                -4.481319657643e-20,
                -5.178184609861e-20,
                -8.344864391643e-20,
                -7.523864637391e-22,
                -2.352041714355e-19,
            ],
            [
                -1.776873413902e-21,
                1.089732939299e-19,
                5.807961380758e-04,
                3.330505796143e-19,
                -1.731384880743e-20,
                1.103241243838e-19,
                6.746061175080e-20,
                3.141075538595e-20,
                -6.904227320354e-20,
            ],
            [
                -9.100340154061e-22,
                1.523011171239e-19,
                3.330505796143e-19,
                5.807961380758e-04,
                8.805313693572e-20,
                -2.167228714023e-19,
                2.876713664998e-19,
                7.934435957849e-20,
                -1.736983034661e-19,
            ],
            [
                4.536462608839e-36,
                -4.481319657643e-20,
                -1.731384880743e-20,
                8.805313693572e-20,
                4.055133126971e-04,
                -5.964807515400e-20,
                -3.134085779095e-21,
                -3.485868720781e-20,
                1.809465268224e-21,
            ],
            [
                -4.056738113699e-37,
                -5.178184609861e-20,
                1.103241243838e-19,
                -2.167228714023e-19,
                -5.964807515400e-20,
                4.055133126971e-04,
                1.661922801254e-21,
                1.036752907397e-19,
                -9.595115766753e-22,
            ],
            [
                2.822847908886e-05,
                -8.344864391643e-20,
                6.746061175080e-20,
                2.876713664998e-19,
                -3.134085779095e-21,
                1.661922801254e-21,
                2.330308752984e-04,
                -1.923260224409e-22,
                -1.144358992193e-04,
            ],
            [
                -9.181931846658e-25,
                -7.523864637391e-22,
                3.141075538595e-20,
                7.934435957849e-20,
                -3.485868720781e-20,
                1.036752907397e-19,
                -1.923260224409e-22,
                2.991004725176e-04,
                -1.736027227993e-21,
            ],
            [
                4.889316000230e-05,
                -2.352041714355e-19,
                -6.904227320354e-20,
                -1.736983034661e-19,
                1.809465268224e-21,
                -9.595115766753e-22,
                -1.144358992193e-04,
                -1.736027227993e-21,
                1.008916808599e-04,
            ],
        ]
    )

    # Analytic
    EI = bigEI(Ey, t)

    # Numerical
    EI_num = bigEI_numerical(Ey, t)

    assert np.allclose(EI, EI_num)
