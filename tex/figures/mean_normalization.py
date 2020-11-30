import starry
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


class Star(object):
    def __init__(self, nlon=300, ydeg=30, linear=True):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.zeros_like(self.lat)
        self.linear = linear

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

    def add_spot(self, lon, lat, radius, contrast):
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        if self.linear:
            self.intensity[idx] -= contrast
        else:
            self.intensity[idx] = -contrast

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
        return 1.0 + self.map.flux(theta=360.0 / period * t)


t = np.linspace(0.5, 3.5, 1000)

# Dark equatorial spot
star1 = Star()
star1.add_spot(0.0, 0.0, 20.0, 1.0)
flux1 = star1.flux(t)

# Less dark equatorial spot w/ polar spot
star2 = Star()
star2.add_spot(0.0, -90.0, 60.0, 0.3925)
star2.add_spot(0.0, 0.0, 20.0, 0.5)
flux2 = star2.flux(t)

# Colormap range
image = star1.map.render()
norm = Normalize(vmin=np.nanmin(image), vmax=np.nanmax(image))

# Plot
fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
ax[0].plot(t - 0.5, flux1, "C0-", lw=2.5)
ax[0].plot(t - 0.5, flux2, "C1--", lw=2.5)
ax[0].set_ylabel("true flux [fractional]", fontsize=20, labelpad=20)
ax[1].plot(t - 0.5, 1e3 * (flux1 / flux1.max() - 1), "C0-", lw=2.5)
ax[1].plot(t - 0.5, 1e3 * (flux2 / flux2.max() - 1), "C1--", lw=2.5)
ax[1].set_ylabel("observed flux [ppt]", fontsize=20)
ax[1].set_xlabel("rotations", fontsize=20)

# Legend
ax1a = plt.axes([0.9, 0.55, 0.2, 0.2])
star1.map.show(ax=ax1a, norm=norm)
ax2a = plt.axes([0.9, 0.25, 0.2, 0.2])
star2.map.show(ax=ax2a, norm=norm)
ax1b = plt.axes([1.1, 0.55, 0.07, 0.2])
ax1b.set_ylim(0, 1)
ax1b.set_xlim(0, 1)
ax1b.plot([0, 1], [0.5, 0.5], "C0-", lw=3)
ax1b.axis("off")
ax2b = plt.axes([1.1, 0.25, 0.07, 0.2])
ax2b.set_ylim(0, 1)
ax2b.set_xlim(0, 1)
ax2b.plot([0, 1], [0.5, 0.5], "C1--", lw=3)
ax2b.axis("off")

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
