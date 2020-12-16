import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import beta as B
from scipy.special import hyp2f1
from sympy.physics.quantum.spin import Rotation
from sympy import re
from tqdm import tqdm
import os


def factorial(n):
    """Just the factorial function, no frills."""
    return Gamma(n + 1)


def clmmpi(l, m, mp, i):
    """The Wigner-d matrix coefficients c^l_{m, m', i}."""
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


def dlmmp(l, m, mp, beta):
    """The Wigner-d matrix term d^l_{m,m'}(beta)."""
    d = 0
    for i in range(2 * l + 1):
        d += (
            clmmpi(l, m, mp, i)
            * np.sign(np.sin(beta)) ** i
            * (1 - np.cos(beta)) ** ((2 * l - i) / 2)
            * (1 + np.cos(beta)) ** (i / 2)
        )
    return d


def dlmmp_sympy(l, m, mp, beta):
    """
    The Wigner-d matrix term d^l_{m,m'}(beta), evaluated using `sympy`.

    Note that the transformation in `sympy` is a **passive** transformation,
    (i.e., it is a rotation of the coordinate system, not of actual points
    in 3D space), so we flip the sign of the angle of rotation when 
    comparing to our implementation.
    """
    return np.float(re(Rotation.d(l, m, mp, -beta).doit().evalf()))


def test_dlmmp(lmax=3, nbeta=10):
    """
    Compare our expression to the one in `sympy` up to degree `lmax`
    for a few different values of `beta`.
    """
    np.random.seed(0)
    for n in tqdm(range(nbeta), disable=bool(int(os.getenv("NOTQDM", "0")))):
        beta = np.random.uniform(-np.pi, np.pi)
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                for mp in range(-l, l + 1):
                    assert np.allclose(
                        dlmmp(l, m, mp, beta), dlmmp_sympy(l, m, mp, beta)
                    )
