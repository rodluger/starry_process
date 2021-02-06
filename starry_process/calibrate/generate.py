from .defaults import update_with_defaults
import starry
import numpy as np
from tqdm import tqdm
import os

starry.config.quiet = True


class Star(object):
    def __init__(
        self,
        nlon=300,
        ydeg=30,
        linear=True,
        smoothing=0.1,
        eps=1e-12,
        u=[0.0, 0.0],
    ):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.zeros_like(self.lat)
        self.linear = linear

        # Instantiate a starry map
        self.map = starry.Map(ydeg=ydeg, udeg=2, lazy=False)
        self.map[1:] = u

        # cos(lat)-weighted SHT
        w = np.cos(self.lat.flatten() * np.pi / 180)
        P = self.map.intensity_design_matrix(
            lat=self.lat.flatten(), lon=self.lon.flatten()
        )
        PTSinv = P.T * (w ** 2)[None, :]
        self.Q = np.linalg.solve(PTSinv @ P + eps * np.eye(P.shape[1]), PTSinv)
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            self.Q *= s[:, None]

    def _angular_distance(self, lam1, lam2, phi1, phi2):
        # https://en.wikipedia.org/wiki/Great-circle_distance
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

    def reset(self):
        self.intensity = np.zeros_like(self.lat)

    def add_spot(self, lon, lat, radius, contrast):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        if self.linear:
            self.intensity[idx] -= contrast
        else:
            self.intensity[idx] = -contrast

    def flux(self, t, period=1.0, inc=60.0):
        # Expand in Ylms
        self.map[:, :] = self.Q @ self.intensity.flatten()
        self.map.inc = inc
        return self.map.flux(theta=360.0 / period * t)


def generate(**kwargs):
    """
    Generate a synthetic ensemble of light curves with similar spot properties.

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    seed = kwargs["seed"]
    gen_kwargs = kwargs["generate"]
    normalized = gen_kwargs["normalized"]
    normalization_method = gen_kwargs["normalization_method"]
    nlon = gen_kwargs["nlon"]
    ydeg = gen_kwargs["ydeg"]
    u = gen_kwargs["u"]
    smoothing = gen_kwargs["smoothing"]
    nlc = gen_kwargs["nlc"]
    npts = gen_kwargs["npts"]
    tmax = gen_kwargs["tmax"]
    period = gen_kwargs["period"]
    ferr = gen_kwargs["ferr"]
    nspots = lambda: max(
        1,
        int(
            gen_kwargs["nspots"]["mu"]
            + gen_kwargs["nspots"]["sigma"] * np.random.randn()
        ),
    )
    radius = lambda: (
        max(
            1.0,
            gen_kwargs["radius"]["mu"]
            + gen_kwargs["radius"]["sigma"] * np.random.randn(),
        )
    )
    longitude = lambda: np.random.uniform(-180, 180)
    if np.isinf(gen_kwargs["latitude"]["sigma"]):
        # Uniform
        latitude = (
            lambda: 180 / np.pi * np.arccos(2 * np.random.random() - 1) - 90
        )
    else:
        latitude = lambda: (
            (1 if np.random.random() < 0.5 else -1)
            * min(
                90,
                max(
                    0,
                    gen_kwargs["latitude"]["mu"]
                    + gen_kwargs["latitude"]["sigma"] * np.random.randn(),
                ),
            )
        )
    contrast = lambda: (
        gen_kwargs["contrast"]["mu"]
        + gen_kwargs["contrast"]["sigma"] * np.random.randn()
    )

    # Generate `nlc` light curves
    np.random.seed(seed)
    t = np.linspace(0, tmax, npts)
    flux0 = np.empty((nlc, npts))
    flux = np.empty((nlc, npts))
    incs = 180 / np.pi * np.arccos(np.random.uniform(0, 1, size=nlc))
    y = np.zeros((nlc, (ydeg + 1) ** 2))
    star = Star(
        nlon=nlon,
        ydeg=ydeg,
        linear=gen_kwargs["nspots"]["linear"],
        smoothing=smoothing,
        u=u,
    )
    for k in tqdm(range(nlc), disable=bool(int(os.getenv("NOTQDM", "0")))):

        # Generate the stellar map
        star.reset()
        nspots_cur = nspots()
        for _ in range(nspots_cur):
            star.add_spot(longitude(), latitude(), radius(), contrast())

        # Get the light curve
        flux0[k] = star.flux(t, period=period, inc=incs[k])

        # Store the coefficients
        y[k] = np.array(star.map.amp * star.map.y)

    # Add photon noise & optionally normalize
    for k in tqdm(range(nlc), disable=bool(int(os.getenv("NOTQDM", "0")))):

        if normalized:

            if normalization_method.lower() == "median":
                flux[k] = (1 + flux0[k]) / (1 + np.median(flux0[k])) - 1
            elif normalization_method.lower() == "mean":
                flux[k] = (1 + flux0[k]) / (1 + np.mean(flux0[k])) - 1
            else:
                raise ValueError("Unknown normalization method.")

            flux[k] += ferr * np.random.randn(npts)

        else:

            flux[k] = flux0[k] + ferr * np.random.randn(npts)

    # Return a dict
    data = dict(
        t=t, flux0=flux0, flux=flux, ferr=ferr, period=period, incs=incs, y=y
    )
    return data
