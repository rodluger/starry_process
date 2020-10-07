from starry_process.design import FluxDesignMatrix
from starry_process.defaults import defaults
import numpy as np
import pytest
import starry


def test_design(ydeg=5, inc=defaults["inc"], period=defaults["period"]):

    # Get the SP design matrix
    t = np.linspace(-1, 1, 50)
    F = FluxDesignMatrix(ydeg)
    A = F(t, period=period, inc=inc).eval()

    # Compare to the starry design matrix
    theta = 360.0 / period * t
    map = starry.Map(ydeg, inc=inc)
    A_starry = map.design_matrix(theta=theta)
    assert np.allclose(A, A_starry)
