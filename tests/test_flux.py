from starry_process.flux import FluxDesignMatrix
from starry_process.defaults import defaults
import numpy as np
import pytest

try:
    import starry

    starry.config.lazy = False

    skip = False
except:
    skip = True


@pytest.mark.skipif(skip, reason="unable to import starry")
def test_flux(ydeg=5, inc=defaults["inc"], period=defaults["inc"]):

    # Get the SP design matrix
    t = np.linspace(-1, 1, 50)
    F = FluxDesignMatrix(ydeg, period=period, inc=inc)
    A = F(t).eval()

    # Compare to the starry design matrix
    theta = 360.0 / period * t
    map = starry.Map(ydeg, inc=inc)
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
