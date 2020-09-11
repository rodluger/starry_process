from starry_process import StarryProcess
import starry
from scipy.stats import norm as Normal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


starry.config.lazy = False


def lat2y(lat):
    # Returns the fractional y position (in [0, 1])
    # corresponding to a given latitude on a Mollweide grid
    lat = lat * np.pi / 180
    theta = lat
    niter = 100
    for n in range(niter):
        theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * theta)
        )
    return np.sin(theta)


class Star(object):
    def __init__(self, nlon=300, ydeg=30):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.ones_like(self.lat)

        # Instantiate a starry map
        self.map = starry.Map(ydeg)

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
        self.intensity = np.ones_like(self.lat)

    def add_spot(self, lon, lat, radius, contrast):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        self.intensity[idx] = 1 - contrast

    def flux(self, t, period=1.0, inc=60.0, smoothing=0.1):
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
        flux = self.map.flux(theta=360.0 / period * t)

        # Median-normalize it
        flux /= np.median(flux)

        return flux


# Discretization settings
nlon = 300
ydeg = 30
smoothing = 0.1

# Light curve settings
nlc = 10
npts = 1000
tmax = 4.0
t = np.linspace(0, tmax, npts)
period = 1.0
ferr = 5e-3

# Spot settings
nspots = 10
rmu = 20.0
rsig = 5.0
lmu = 30.0
lsig = 5.0
cmu = 0.75
csig = 0.10

# Generate `nlc` light curves
np.random.seed(0)
star = Star(nlon=nlon, ydeg=ydeg)
flux0 = np.empty((nlc, npts))
images = [None for k in range(nlc)]
incs = np.zeros(nlc)
for k in tqdm(range(nlc)):

    # Inclination prior is p(i) di = sin(i) di, so...
    incs[k] = np.arccos(np.random.uniform(0, 1)) * 180 / np.pi

    # Generate the stellar map
    star.reset()
    for n in range(nspots):
        radius = Normal.rvs(rmu, rsig)
        sign = 1 if np.random.random() < 0.5 else -1
        lat = sign * Normal.rvs(lmu, lsig)
        lon = np.random.uniform(-180, 180)
        contrast = Normal.rvs(cmu, csig)
        star.add_spot(lon, lat, radius, contrast)

    # Get the light curve
    flux0[k] = star.flux(t, period=period, inc=incs[k], smoothing=smoothing)

    # Render the surface
    images[k] = star.map.render(projection="moll")

# Add photon noise
flux = flux0 + ferr * np.random.randn(*flux0.shape)

# Save the data
np.savez("ensemble_data.npz", t=t, period=period, flux=flux, ferr=ferr)

# Plot the data
fig, ax = plt.subplots(2, nlc, figsize=(16, 2))
vmin = 0
vmax = 1
yrng = 1.1 * np.max(np.abs(1e3 * (flux0 - 1)))
ymin = -yrng
ymax = yrng
xe = 2 * np.linspace(-1, 1, 1000)
ye = np.sqrt(1 - (0.5 * xe) ** 2)
eps = 0.01
xe = (1 - eps) * xe
ye = (1 - 0.5 * eps) * ye
for k in range(nlc):
    im = ax[0, k].imshow(
        images[k],
        origin="lower",
        extent=(-2, 2, -1, 1),
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
    )
    ax[0, k].plot(xe, ye, "k-", lw=1, clip_on=False)
    ax[0, k].plot(xe, -ye, "k-", lw=1, clip_on=False)
    ax[0, k].plot(0, lat2y(90 - incs[k]), "kx", ms=3)
    ax[0, k].axis("off")
    ax[1, k].plot(t, 1e3 * (flux[k] - 1), "k.", alpha=0.3, ms=1)
    ax[1, k].plot(t, 1e3 * (flux0[k] - 1), "C0-", lw=1)
    ax[1, k].set_ylim(ymin, ymax)
    if k == 0:
        cax = fig.add_axes([0.91, 0.55, 0.005, 0.3])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("intensity", fontsize=8)
        cbar.set_ticks([0, 0.5, 1])
        cbar.ax.tick_params(labelsize=6)
        ax[1, k].spines["top"].set_visible(False)
        ax[1, k].spines["right"].set_visible(False)
        ax[1, k].set_xlabel("rotations", fontsize=8)
        ax[1, k].set_ylabel("flux [ppt]", fontsize=8)
        ax[1, k].set_xticks([0, 1, 2, 3, 4])
        for tick in (
            ax[1, k].xaxis.get_major_ticks() + ax[1, k].yaxis.get_major_ticks()
        ):
            tick.label.set_fontsize(6)
    else:
        ax[1, k].axis("off")

# Save the figure
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight", dpi=300)
