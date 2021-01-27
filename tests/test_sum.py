from starry_process import StarryProcess, gauss2beta
import numpy as np
import pytest

starry = pytest.importorskip("starry")


def test_sample(tol=5):

    # Instantiate the two GPs
    a, b = gauss2beta(45, 1)
    sp1 = StarryProcess(r=10, a=a, b=b)
    a, b = gauss2beta(0, 1)
    sp2 = StarryProcess(r=10, a=a, b=b)

    # Sum them and draw a sample
    sp = sp1 + sp2
    y = sp.sample_ylm().eval()

    # Instantiate a starry map to compute
    # the longitudinally-averaged intensity
    map = starry.Map(15, lazy=False)
    map[:, :] = y
    lat = np.linspace(-90, 90, 300)
    lon = np.linspace(-180, 180, 600)
    lat_, lon_ = np.meshgrid(lat, lon)
    lat_ = lat_.flatten()
    lon_ = lon_.flatten()
    I = np.mean(
        map.intensity(lat=lat_, lon=lon_).reshape(len(lon), len(lat)), axis=0
    )

    # Get the 3 lowest local minima
    grad = np.gradient(I)
    idx = (grad[1:] > 0) & (grad[:-1] < 0)
    k = np.argsort(I[:-1][idx])[:3]
    min_lats = np.sort(lat[:-1][idx][k])

    # Now check that these are about (-45, 0, 45)
    assert np.abs(min_lats[0] - (-45)) < tol
    assert np.abs(min_lats[1]) < tol
    assert np.abs(min_lats[2] - 45) < tol


def test_likelihood():
    t = np.linspace(0, 1, 100)
    flux = np.random.randn(100)
    sp = StarryProcess(r=10) + StarryProcess(r=20)
    ll = sp.log_likelihood(t, flux, 1.0).eval()
    assert np.isfinite(ll)
