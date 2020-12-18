import numpy as np
from starry_process.ops import LOp
import starry


def test_paper_expression(ntests=10):
    """
    Check that the expression in the paper agrees with the
    expression in the code.

    """
    # Internal expression
    L = LOp(ydeg=1, udeg=1)

    # Expression in the paper
    def L_paper(u1):
        return np.array(
            [
                [1 - u1, 0, u1 / np.sqrt(3), 0],
                [0, 1 - u1, 0, 0],
                [u1 / np.sqrt(3), 0, 1 - u1, 0],
                [0, 0, 0, 1 - u1],
                [0, 0, 0, 0],
                [0, u1 / np.sqrt(5), 0, 0],
                [0, 0, 2 * u1 / np.sqrt(15), 0],
                [0, 0, 0, u1 / np.sqrt(5)],
                [0, 0, 0, 0],
            ]
        ) / (1 - u1 / 3)

    np.random.seed(0)
    for u1 in np.random.random(ntests):
        assert np.allclose(L([u1]).eval(), L_paper(u1))


def test_ld():
    """
    Show that our expression for limb darkening agrees
    with the definition.

    """
    # A fiducial quadratic limb darkening model
    u = [0.5, 0.25]
    L = LOp(ydeg=5, udeg=2)(u).eval()

    # Pick random points on the unit disk
    npts = 100
    r = np.random.random(npts)
    theta = 2 * np.pi * np.random.random(npts)
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    z = np.sqrt(1 - x ** 2 - y ** 2)

    # Convert to lat/lon
    lat = (0.5 * np.pi - np.arccos(y)) * 180 / np.pi
    lon = (np.arctan2(x, z)) * 180 / np.pi

    # Get the matrix `P` that transforms spherical harmonic
    # coefficients to intensity at points on the disk
    P = starry.Map(7, lazy=False).intensity_design_matrix(lat=lat, lon=lon)

    # Random degree 5 map
    y = np.append([1.0], 0.025 * np.random.randn(35))

    # Compute the intensity without limb darkening
    I = P[:, :36] @ y

    # Compute the intensity with limb darkening
    # using our linear operator
    I_ld = P @ L @ y

    # Compute the intensity with limb darkening
    # via the definition
    f = 1 - u[0] * (1 - z) - u[1] * (1 - z) ** 2
    I_ld_def = I * f

    # Now show that our expression is the same as what
    # we get from the definition (up to a multiplicative constant)
    assert np.allclose(I_ld / I_ld_def, I_ld[0] / I_ld_def[0])
