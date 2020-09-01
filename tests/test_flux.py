from starry_process.flux import FluxDesignMatrix
import numpy as np
import pytest

try:
    import starry

    skip = False
except:
    skip = True


@pytest.mark.skipif(skip, reason="unable to import starry")
def test_flux(ydeg=5, inc=60.0, period=1.0):

    # Get the SP design matrix
    t = np.linspace(-1, 1, 50)
    F = FluxDesignMatrix(ydeg)
    F.set_params(period, inc)
    A = F(t).eval()

    # Compare to the starry design matrix
    theta = 360.0 / period * t
    map = starry.Map(ydeg, lazy=False, inc=inc)
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
