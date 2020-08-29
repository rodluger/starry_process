from starry_process.flux import FluxDesignMatrix
import numpy as np
import pytest

try:
    import starry

    skip = False
except:
    skip = True


@pytest.mark.skipif(skip, reason="unable to import starry")
def test_flux(ydeg=5, inc=60.0):

    # Get the SP design matrix
    theta = np.linspace(-360, 360, 50)
    A = FluxDesignMatrix(ydeg)(theta * np.pi / 180, inc * np.pi / 180).eval()

    # Compare to the starry design matrix
    map = starry.Map(ydeg, lazy=False, inc=inc)
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
