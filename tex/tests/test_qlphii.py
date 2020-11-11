import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gamma
from scipy.special import beta as B
from scipy.special import hyp2f1


def integrand_phi(phi, alpha, beta, l, i):
    sin = np.sin(phi)
    cos = np.cos(phi)
    return (
        0.5
        * np.sign(sin) ** i
        * np.abs(sin)
        * cos ** (alpha - 1)
        * (1 - cos) ** (l + beta - i / 2 - 1)
        * (1 + cos) ** (i / 2)
    )


def qli_numerical_phi(l, i, alpha, beta):
    return quad(
        integrand_phi, -0.5 * np.pi, 0.5 * np.pi, args=(alpha, beta, l, i)
    )[0]


def integrand_x(x, alpha, beta, l, i):
    return (
        x ** (alpha - 1)
        * (1 - x) ** (l + beta - i / 2 - 1)
        * (1 + x) ** (i / 2)
    )


def qli_numerical_x(l, i, alpha, beta):
    return quad(integrand_x, 0, 1, args=(alpha, beta, l, i))[0]


def qli_closed_form(l, i, alpha, beta):
    return B(alpha, beta + l - i / 2) * hyp2f1(
        -i / 2, alpha, l + alpha + beta - i / 2, -1
    )


def test_qli(
    lmax=10, alpha=3.0, beta=5.0,
):
    for l in range(lmax + 1):
        for i in range(0, 2 * l + 1, 2):
            assert np.allclose(
                qli_closed_form(l, i, alpha, beta),
                [
                    qli_numerical_phi(l, i, alpha, beta),
                    qli_numerical_x(l, i, alpha, beta),
                ],
            )
