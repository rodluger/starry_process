import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gamma


def integrand(phi, l, i):
    sin = np.sin(phi)
    cos = np.cos(phi)
    return (
        np.sign(sin) ** i * (1 - cos) ** (l - i / 2) * (1 + cos) ** (i / 2)
    ) / (2 * np.pi)


def qli_numerical(l, i):
    return quad(integrand, -np.pi, np.pi, args=(l, i))[0]


def qli_closed_form(l, i):
    return (
        2 ** l
        / np.pi
        * Gamma(0.5 * (1 + i))
        * Gamma(l + 0.5 * (1 - i))
        / Gamma(l + 1)
    )


def test_qli(lmax=10):
    for l in range(lmax + 1):
        for i in range(0, 2 * l + 1, 2):
            assert np.allclose(qli_closed_form(l, i), qli_numerical(l, i),)
