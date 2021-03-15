from starry_process.flux import FluxIntegral
from starry_process.defaults import defaults
from starry_process.compat import theano, tt
import numpy as np
import pytest

starry = pytest.importorskip("starry")


def test_design(ydeg=15, i=defaults["i"], p=defaults["p"], u=defaults["u"]):

    # Get the SP design matrix
    t = np.linspace(-1, 1, 50)
    N = (ydeg + 1) ** 2
    F = FluxIntegral(
        tt.as_tensor_variable(np.zeros(N)),
        tt.as_tensor_variable(np.zeros((N, N))),
        ydeg=ydeg,
        marginalize_over_inclination=False,
    )
    A = F.design_matrix(t, i, p, u).eval()

    # Compare to the starry design matrix
    theta = 360.0 / p * t
    map = starry.Map(ydeg, udeg=len(u), inc=i)
    map[1:] = u
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
