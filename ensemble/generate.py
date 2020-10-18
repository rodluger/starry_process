from starry_process import StarryProcess
from starry_process.wigner import R
from starry_process.size import Spot
import starry
from scipy.stats import norm as Normal
import numpy as np
from tqdm import tqdm
import os
import json


PATH = os.path.abspath(os.path.dirname(__file__))


class Star(object):
    def __init__(self, nlon=300, ydeg=30):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.zeros_like(self.lat)

        # Instantiate a starry map
        self.map = starry.Map(ydeg, lazy=False)

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

    def add_spot(self, lon, lat, radius, contrast, linear=True):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        if linear:
            self.intensity[idx] -= contrast
        else:
            self.intensity[idx] = -contrast

    def flux(self, t, period=1.0, inc=60.0, smoothing=0.1, normalize=True):
        # Expand in Ylms
        self.map.load(self.intensity)
        self.map.inc = inc

        # Smooth to get rid of ringing
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(self.map.ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            self.map._y *= s

        # Get the flux
        flux = 1 + self.map.flux(theta=360.0 / period * t)

        # Median-normalize it
        if normalize:
            flux /= np.median(flux)

        # Remove the baseline
        flux -= 1

        return flux


def generate(runid):

    # Get input kwargs
    INPUT_FILE = os.path.join(PATH, "{:02d}".format(runid), "input.json")
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("generate", {})
    use_starry_process = inputs.get("use_starry_process", False)
    normalize = inputs.get("normalize", True)
    nlon = inputs.get("nlon", 300)
    ydeg = inputs.get("ydeg", 30)
    smoothing = inputs.get("smoothing", 0.1)
    seed = inputs.get("seed", 0)
    nlc = inputs.get("nlc", 10)
    npts = inputs.get("npts", 1000)
    tmax = inputs.get("tmax", 4.0)
    periods = inputs.get("periods", 1.0)
    ferr = inputs.get("ferr", 1e-4)
    nspots = inputs.get("nspots", 20)
    rmu = inputs.get("rmu", 20.0)
    rsig = inputs.get("rsig", 1.0)
    lmu = inputs.get("lmu", 30.0)
    lsig = inputs.get("lsig", 5.0)
    cmu = inputs.get("cmu", 0.05)
    csig = inputs.get("csig", 0.0)

    # Convert to standard params
    sp = StarryProcess()
    r = rmu
    l = lmu
    try:
        a, b = sp.latitude._transform.transform(lmu, lsig)
        assert not (np.isnan(a) or np.isnan(b))
    except:
        a = 0.5
        b = 0.5
    c = cmu
    n = float(nspots)

    # Generate `nlc` light curves
    np.random.seed(seed)
    t = np.linspace(0, tmax, npts)
    flux0 = np.empty((nlc, npts))
    flux = np.empty((nlc, npts))
    images = [None for k in range(nlc)]
    incs = 180 / np.pi * np.arccos(np.random.uniform(0, 1, size=nlc))
    periods = periods * np.ones(nlc)

    if use_starry_process:

        y = np.zeros((nlc, (ydeg + 1) ** 2))
        map = starry.Map(ydeg, lazy=False)
        for k in tqdm(range(nlc)):

            # Generate the stellar map
            for _ in range(nspots):
                radius = Normal.rvs(rmu, rsig)
                sign = 1 if np.random.random() < 0.5 else -1
                lat = sign * Normal.rvs(lmu, lsig)
                lon = np.random.uniform(-180, 180)
                contrast = Normal.rvs(cmu, csig)
                y0 = (
                    np.pi
                    * contrast
                    * Spot(ydeg=ydeg).get_y(radius * np.pi / 180).eval()
                )
                Rx = R(
                    ydeg,
                    phi=lat * np.pi / 180.0,
                    cos_alpha=0,
                    sin_alpha=1,
                    cos_gamma=0,
                    sin_gamma=-1,
                )
                Ry = R(
                    ydeg,
                    phi=lon * np.pi / 180.0,
                    cos_alpha=1,
                    sin_alpha=0,
                    cos_gamma=1,
                    sin_gamma=0,
                )
                for l in range(ydeg + 1):
                    i = slice(l ** 2, (l + 1) ** 2)
                    y[k, i] += Ry[l] @ (Rx[l] @ y0[i])

            map[:, :] = y[k]
            images[k] = 1.0 + map.render(projection="moll", res=300)

            map.inc = incs[k]
            flux0[k] = 1 + map.flux(theta=360.0 / periods[k] * t)
            if normalize:
                flux0[k] /= np.median(flux0[k])
            flux0[k] -= 1

    else:

        y = np.zeros((nlc, (ydeg + 1) ** 2))
        star = Star(nlon=nlon, ydeg=ydeg)
        for k in tqdm(range(nlc)):

            # Generate the stellar map
            star.reset()
            for _ in range(nspots):
                radius = Normal.rvs(rmu, rsig)
                sign = 1 if np.random.random() < 0.5 else -1
                lat = sign * Normal.rvs(lmu, lsig)
                lon = np.random.uniform(-180, 180)
                contrast = Normal.rvs(cmu, csig)
                star.add_spot(lon, lat, radius, contrast)

            # Get the light curve
            flux0[k] = star.flux(
                t,
                period=periods[k],
                inc=incs[k],
                smoothing=smoothing,
                normalize=normalize,
            )

            # Render the surface
            images[k] = 1.0 + star.map.render(projection="moll", res=300)
            y[k] = np.array(star.map.amp * star.map.y)

    # Add photon noise
    for k in tqdm(range(nlc)):
        flux[k] = flux0[k] + ferr * np.random.randn(npts)

    # Return dicts
    truth = dict(
        r=r,
        a=a,
        b=b,
        c=c,
        n=n,
        periods=periods,
        incs=incs,
        nspots=nspots,
        rmu=rmu,
        rsig=rsig,
        lmu=lmu,
        lsig=lsig,
        cmu=cmu,
        csig=csig,
        images=images,
        y=y,
    )
    data = dict(t=t, flux0=flux0, flux=flux, ferr=ferr)

    return data, truth
