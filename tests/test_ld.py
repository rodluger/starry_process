from starry_process.ops import rTA1LOp
from starry_process import StarryProcess
import numpy as np
from theano.configparser import change_flags
from starry_process.compat import theano, tt
import pytest

try:
    import starry
except ImportError:
    starry = None


def test_rTA1L_grad():
    with change_flags(compute_test_value="off"):
        op = rTA1LOp(ydeg=1, udeg=3)
        theano.gradient.verify_grad(
            lambda u1, u2, u3: op([u1, u2, u3]),
            [0.5, 0.25, 0.1],
            n_tests=1,
            rng=np.random,
        )


@pytest.mark.skipif(starry is None, reason="starry not installed")
def test_compare_to_starry(ydeg=15, udeg=2):

    np.random.seed(0)
    u = np.random.random(udeg)
    y = np.random.randn((ydeg + 1) ** 2 - 1)

    # This
    op = rTA1LOp(ydeg=ydeg, udeg=udeg)
    flux1 = op(u).eval() @ np.concatenate(((1,), y))

    map = starry.Map(ydeg=ydeg, udeg=udeg)
    map[1:] = u
    map[1:, :] = y
    flux2 = map.flux()

    assert np.allclose(flux1, flux2)


def test_null_limb_darkening():

    t = np.linspace(0, 1, 300)
    cov1 = StarryProcess(udeg=0).cov(t).eval()
    cov2 = StarryProcess(udeg=2).cov(t, u=[0.0, 0.0]).eval()
    assert np.allclose(cov1, cov2)
