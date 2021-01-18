import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import starry
from scipy.linalg import svd
import os


class Star(object):
    def __init__(
        self, nlon=300, ydeg=30, eps=1e-12, smoothing=0.1, inc=60, npts=1000
    ):
        # Generate a uniform intensity grid
        self.nlon = nlon
        self.nlat = nlon // 2
        self.lon = np.linspace(-180, 180, self.nlon)
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon, self.lat = np.meshgrid(self.lon, self.lat)
        self.intensity = np.zeros_like(self.lat)

        # Instantiate a starry map
        self.map = starry.Map(ydeg, inc=inc, lazy=False)

        # Null space operators
        t = np.linspace(0, 1, npts, endpoint=False)
        self.A = self.map.design_matrix(theta=360 * t)
        rank = np.linalg.matrix_rank(self.A)
        _, _, VT = svd(self.A)
        self.N = VT[rank:].T @ VT[rank:]  # null space operator
        self.R = VT[:rank].T @ VT[:rank]  # row space operator

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

    def add_spot(self, lon=None, lat=None, radius=20, contrast=0.1):
        if lon is None:
            lon = np.random.uniform(-180, 180)
        if lat is None:
            lat = np.arccos(np.random.random()) * 180 / np.pi
            lat *= np.sign(np.random.uniform(-1, 1))
        idx = self._angular_distance(lon, self.lon, lat, self.lat) <= radius
        self.intensity[idx] -= contrast

    def get_y(self):
        return self.Q @ self.intensity.flatten()

    def get_y_null(self):
        return self.N @ self.get_y()

    def get_y_image(self):
        return self.R @ self.get_y()


# Number of phases to show
phases = 12
theta = np.linspace(-180, 180, phases, endpoint=True)

# Colormap normalization
norm = Normalize(vmin=-0.08, vmax=0.03)

# Properties of the single spot that's
# not (fully) in the null space
radius0 = 20
lon0 = 0
lat0 = 30
contrast0 = 0.1

# Properties of the other spots, which
# are _entirely_ in the null space
lon = [80, 180, -120, -30, -80]
lat = [-10, 45, 0, -20, 10]
radius = [20, 15, 20, 15, 20]
contrast = [0.15, 0.1, 0.15, 0.15, 0.15]

# Set up
N = len(lon)
fig = plt.figure(figsize=(24, 16))
lw = np.linspace(1, 5, N + 1)[::-1]
t = np.linspace(0, 1, 1000)

ax = np.array(
    [
        [plt.subplot2grid((N + 4, phases), (i, j)) for j in range(phases)]
        for i in range(N + 1)
    ]
)
ax_lc = plt.subplot2grid(
    (N + 4, phases), (N + 1, 0), colspan=phases, rowspan=3
)

# Get the Ylm expansion of the main spot and visualize the map
star = Star()
star.add_spot(lon=lon0, lat=lat0, radius=radius0, contrast=contrast0)
y0 = star.get_y()

# Add some inhomogeneities for fun
np.random.seed(1)
alpha = 0.0075
lam = 0.25
l0 = 5
ybkg = np.concatenate(
    [
        alpha * np.exp(-lam * (l - l0) ** 2) * np.random.randn(2 * l + 1)
        for l in range(31)
    ]
)
ybkg[0] = 0
y0 += ybkg
star.map[:, :] = y0

for k in range(phases):
    star.map.show(ax=ax[0, k], theta=theta[k], norm=norm)
flux = star.map.flux(theta=np.linspace(-180, 180, 1000))
flux -= np.median(flux)
flux *= 1e3
ax_lc.plot(t, flux, lw=lw[0], label="{}".format(1))

# Get their Ylm expansions and visualize their maps
for n in range(N):
    print(n)
    star.add_spot(
        lon=lon[n], lat=lat[n], radius=radius[n], contrast=contrast[n]
    )
    y = star.get_y_null() + y0
    star.map[:, :] = y
    for k in range(phases):
        star.map.show(ax=ax[n + 1, k], theta=theta[k], norm=norm)

    # Plot the light curve
    flux = star.map.flux(theta=np.linspace(-180, 180, 1000))
    flux -= np.median(flux)
    flux *= 1e3
    ax_lc.plot(t, flux, lw=lw[n], label="{}".format(n + 2))

# Appearance
for axis in ax.flatten():
    axis.set_rasterization_zorder(2)
for n, axis in enumerate(ax[:, 0]):
    axis.annotate(
        "{} spot{}".format(n + 1, "" if n == 0 else "s"),
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-10, 0),
        ha="right",
        va="center",
        textcoords="offset points",
        fontsize=18,
    )
ax_lc.set_xlabel("rotation phase", fontsize=28)
ax_lc.set_ylabel("flux [ppt]", fontsize=28)
ax_lc.margins(0.035, None)
ax_lc.legend(loc="lower right", fontsize=18, title="number of spots")
for tick in ax_lc.xaxis.get_major_ticks():
    tick.label.set_fontsize(22)
for tick in ax_lc.yaxis.get_major_ticks():
    tick.label.set_fontsize(22)

fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
