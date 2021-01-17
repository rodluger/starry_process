import numpy as np


def angular_distance(lam1, lam2, phi1, phi2):
    """
    Angular distance between points at lon-lat coordinates
    (lam1, phi1) and (lam2, phi2) in degrees.

    See https://en.wikipedia.org/wiki/Great-circle_distance
    """
    return (
        np.arccos(
            np.sin(phi1 * np.pi / 180) * np.sin(phi2 * np.pi / 180)
            + np.cos(phi1 * np.pi / 180)
            * np.cos(phi2 * np.pi / 180)
            * np.cos((lam2 - lam1) * np.pi / 180)
        )
        * 180
        / np.pi
    )


def fS_samp(r, N=100000):
    """
    Fractional spot coverage area computed by sampling a bunch
    of points on the sphere and computing the fraction that are
    inside a spot of radius `r`.

    See https://mathworld.wolfram.com/SpherePointPicking.html
    """
    u = np.random.random(N)
    v = np.random.random(N)
    lam = 360 * u - 180
    phi = np.arccos(2 * v - 1) * 180 / np.pi - 90
    delta_angle = angular_distance(0, lam, 0, phi)
    return np.array(
        [np.count_nonzero(delta_angle <= r_) / N for r_ in np.atleast_1d(r)]
    )


def fS_exact(r):
    """
    Analytical expression for the fractional spot coverage.

    """
    return 0.5 * (1 - np.cos(r * np.pi / 180))


def test_fS():
    r = np.linspace(0, 45, 100)
    assert np.allclose(fS_samp(r), fS_exact(r), atol=1e-2)
