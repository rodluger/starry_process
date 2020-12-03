import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import starry
from corner import corner as _corner
import os
from scipy.linalg import cho_factor


def corner(*args, **kwargs):
    """
    Override `corner` to make some appearance tweaks.
    
    """
    # Get the usual corner plot
    figure = _corner(*args, **kwargs)

    # Get the axes
    ndim = int(np.sqrt(len(figure.axes)))
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Smaller tick labels
    for ax in axes[1:, 0]:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(formatter)
    for ax in axes[-1, :]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(formatter)

    # Pad the axes to always include the truths
    truths = kwargs.get("truths", None)
    if truths is not None:
        for row in range(1, ndim):
            for col in range(row):
                lo, hi = np.array(axes[row, col].get_xlim())
                if truths[col] < lo:
                    lo = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)
                elif truths[col] > hi:
                    hi = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)

                lo, hi = np.array(axes[row, col].get_ylim())
                if truths[row] < lo:
                    lo = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)
                elif truths[row] > hi:
                    hi = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)

    return figure


class Star(object):
    def __init__(self, nlon=300, ydeg=15, linear=True):
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

    def get_y(self, smoothing=0.1):
        # Expand in Ylms
        self.map.load(self.intensity)

        # Smooth to get rid of ringing
        if smoothing > 0:
            l = np.concatenate(
                [np.repeat(l, 2 * l + 1) for l in range(self.map.ydeg + 1)]
            )
            s = np.exp(-0.5 * l * (l + 1) * smoothing ** 2)
            self.map._y *= s
        return self.map._y * self.map.amp


# Settings
ydeg = 15
nspots = 5
radius = lambda: 20
longitude = lambda: np.random.uniform(-180, 180)
latitude = lambda: (
    (1 if np.random.random() < 0.5 else -1)
    * min(90, max(0, 30 + 5 * np.random.randn(),),)
)
contrast = lambda: 0.1
nmaps = 50000
y_inds = np.array([1, 48, 120, 168, 225])
seed = 0
clobber = False

DATA_FILE = os.path.abspath(__file__.replace(".py", ".npz"))
if clobber or not os.path.exists(DATA_FILE):

    # Sample `nmaps` sph harm vectors
    np.random.seed(0)
    y = np.zeros((nmaps, (ydeg + 1) ** 2))
    star = Star(ydeg=ydeg)
    for k in tqdm(range(nmaps)):
        star.reset()
        for n in range(nspots):
            star.add_spot(longitude(), latitude(), radius(), contrast())
        y[k] = star.get_y()

    # Keep only the most interesting (i.e., non-gaussian) coeffs
    try:
        y = y[:, y_inds]
        np.savez(DATA_FILE, y=y)
    except:
        breakpoint()

else:

    y = np.load(DATA_FILE)["y"]

# Plot the data
l = np.array(np.floor(np.sqrt(y_inds)), dtype=int)
m = y_inds - l ** 2 - l
labels = [r"$y^{{{}}}_{{{}}}$".format(l_, m_) for l_, m_ in zip(l, m)]
fig = corner(y, labels=labels)
xlims = [ax.get_xlim() for ax in np.array(fig.axes).flatten()]
ylims = [ax.get_ylim() for ax in np.array(fig.axes).flatten()]

# Plot the Gaussian approximation
mu = np.mean(y, axis=0)
cov = np.cov(y.T)
L = np.tril(cho_factor(cov, lower=True)[0])
np.random.seed(0)
u = np.random.randn(len(y_inds), 50000)
ygauss = mu.reshape(1, -1) + (L @ u).T
color = lambda i, alpha: "{}{}".format(
    matplotlib.colors.to_hex("C{}".format(i)),
    ("0" + hex(int(alpha * 256)).split("0x")[-1])[-2:],
)
levels = 1.0 - np.exp(-0.5 * np.array([2.0, 1.5, 1.0, 0.5]) ** 2)
alphas = [0.1, 0.2, 0.3, 0.4]
for k in range(4):
    fig = corner(
        ygauss,
        fig=fig,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        no_fill_contours=True,
        color=color(1, alphas[k]),
        contourf_kwargs=dict(),
        contour_kwargs=dict(alpha=0.5),
        bins=20,
        hist_bin_factor=5,
        smooth=2.0,
        hist_kwargs=dict(alpha=0),
        levels=[levels[k]],
    )
[ax.set_xlim(*xlim) for ax, xlim in zip(np.array(fig.axes).flatten(), xlims)]
[ax.set_ylim(*ylim) for ax, ylim in zip(np.array(fig.axes).flatten(), ylims)]

for ax in np.array(fig.axes).flatten():
    for c in ax.collections:
        c.set_rasterized(True)

# We're done
fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight", dpi=300)
