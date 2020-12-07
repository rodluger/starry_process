from starry_process.ops import rTA1LOp
import starry
import numpy as np
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import theano
import theano.tensor as tt
import pytest


def test_rTA1L_grad():
    with change_flags(compute_test_value="off"):
        op = rTA1LOp(ydeg=1, udeg=3)
        verify_grad(
            lambda u1, u2, u3: op([u1, u2, u3]), [0.5, 0.25, 0.1], n_tests=1,
        )


def test_compare_to_starry(ydeg=15, udeg=2):

    np.random.seed(0)
    u = np.random.random(udeg)
    y = np.random.randn((ydeg + 1) ** 2 - 1)

    # This
    op = rTA1LOp(ydeg=ydeg, udeg=udeg)
    flux1 = op(u).eval() @ np.concatenate(((1,), y))

    # Starry
    import starry

    map = starry.Map(ydeg=ydeg, udeg=udeg)
    map[1:] = u
    map[1:, :] = y
    flux2 = map.flux()

    assert np.allclose(flux1, flux2)

