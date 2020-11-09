from starry_process.flux import FluxIntegral
from starry_process.defaults import defaults
import theano.tensor as tt
import numpy as np
import pytest
import starry


def test_design(ydeg=15, i=defaults["i"], p=defaults["p"]):

    # Get the SP design matrix
    t = np.linspace(-1, 1, 50)
    N = (ydeg + 1) ** 2
    F = FluxIntegral(
        tt.as_tensor_variable(np.zeros(N)),
        tt.as_tensor_variable(np.zeros((N, N))),
        ydeg=ydeg,
        marginalize_over_inclination=False,
    )
    A = F._design_matrix(t, i, p).eval()

    # Compare to the starry design matrix
    theta = 360.0 / p * t
    map = starry.Map(ydeg, inc=i)
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
