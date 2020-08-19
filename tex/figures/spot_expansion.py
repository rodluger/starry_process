import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import starry
from scipy.optimize import minimize
from tqdm import tqdm


def hwhm(r):
    """
    Return the theoretical half-width at half minimum as a function of r.
    
    """
    return np.arccos((2 + 3 * r * (2 + r)) / (2 * (1 + r) ** 3)) * 180 / np.pi


def r_max(hwhm_max=60):
    """
    Returns the value of r corresponding to `hwhm_max`.
    
    """
    f = lambda r: (hwhm(r) - hwhm_max) ** 2
    res = minimize(f, 2.0)
    return res.x[0]


def corr(r, c):
    """Intensity correction function."""
    rho = (r - c[0]) / c[1]
    return 1 + c[2] * (1 - rho) ** c[3]


def get_c(ydeg, hwhm_max=60, hwhm_min=15, npts=500):
    """
    Return the coefficients for the radius transformation.

    """
    c = np.zeros(4)

    # Minimum r: we need to optimize numerically
    loss = lambda p: (hwhm_empirical(ydeg, p[0]) - hwhm_min) ** 2
    res = minimize(loss, 0.1526)
    rmin = res.x[0]
    c[0] = rmin

    # Maximum r (easy)
    rmax = r_max(hwhm_max=hwhm_max)
    c[1] = rmax - rmin

    # Now compute the coefficients of the intensity
    # correction, c[2] and c[3].

    # Array over which to compute the loss
    r = np.linspace(rmin + 1e-6, rmax - 1e-6, npts)

    # Get the actual (absolute value of the) intensity at the peak
    l = np.arange(ydeg + 1).reshape(1, -1)
    term = np.sum((1 + r.reshape(-1, 1)) ** -l, axis=-1)
    I = -0.5 * r * (1 - (2 + r) / (1 + r) * term)

    # This is the factor by which we need to normalize the function
    norm = 1.0 / I

    # Find the coefficients of the fit (least squares)
    diff = lambda p: np.sum((norm - corr(r, [c[0], c[1], p[0], p[1]])) ** 2)
    res = minimize(diff, [0.1, 50.0])
    c[2:] = res.x

    return c


def hwhm_empirical(ydeg, r):
    """
    Return the empirical half-width at half minimum as a function of r.
    
    """
    # Setup
    r = np.atleast_1d(r)
    hwhm_empirical = np.zeros_like(r)

    # Legendre expansion
    map = starry.Map(ydeg, lazy=False)
    x = np.zeros(map.Ny)
    l = np.arange(1, map.ydeg + 1)
    for k in range(len(r)):
        x[0] = -0.5 * r[k] * (1 + r[k]) ** -1
        x[l * (l + 1)] = (
            -1.0
            / np.sqrt(2 * l + 1)
            * (
                (1 + r[k]) ** -(l + 1) * r[k]
                + 0.5 * (1 + r[k]) ** -(l + 1) * r[k] ** 2
            )
        )
        map[:, :] = x

        # Find the HWHM
        halfmax = 0.5 * map.intensity(lon=0)

        def loss(theta):
            return (map.intensity(lon=theta) - halfmax) ** 2

        res = minimize(loss, hwhm(max(0.1, r[k])))
        hwhm_empirical[k] = res.x[0]

    return hwhm_empirical


# Settings
ydeg = 15
hwhm_min = 15
hwhm_max = 75
ncurves = 10
cmap = lambda x: plt.get_cmap("plasma")(0.85 * (1 - x))
starry.config.quiet = True

# Get the radius transform coeffs
c = get_c(ydeg, hwhm_max=hwhm_max, hwhm_min=hwhm_min)

# Figure setup
fig = plt.figure(figsize=(12, 10))
ax = [
    plt.subplot2grid((2, 2), (0, 0), colspan=2),
    plt.subplot2grid((2, 2), (1, 0)),
    plt.subplot2grid((2, 2), (1, 1)),
]
fig.subplots_adjust(hspace=0.225)
fig.subplots_adjust(wspace=0.085)


# ---- TOP PANEL ----


# Instantiate the starry map
map = starry.Map(ydeg, lazy=False)
l = np.arange(1, map.ydeg + 1)

# Longitude array
lon = np.linspace(-180, 180, 1000)

# We need to find the values of r corresponding
# to `ncurves` equally spaced values of HWHM
# We'll do nearest-neighbor, since it's fastest
hwhms = np.linspace(hwhm_min, hwhm_max, ncurves)
r_ = np.logspace(-2, 1, 100)
h_ = hwhm_empirical(ydeg, r_)

# Plot the intensity profile for each radius
for k in tqdm(range(ncurves)):

    # The current value of r
    r = r_[np.argmin(np.abs(hwhms[k] - h_))]

    # Legendre expansion
    x = np.zeros(map.Ny)
    x[0] = -0.5 * r * (1 + r) ** -1
    x[l * (l + 1)] = (
        -1.0
        / np.sqrt(2 * l + 1)
        * ((1 + r) ** -(l + 1) * r + 0.5 * (1 + r) ** -(l + 1) * r ** 2)
    )

    # Intensity correction
    x *= corr(r, c)

    # Compute the (unit normalized) intensity
    map[:, :] = x
    I = np.pi * map.intensity(lon=lon)

    # Plot it
    if k == 0:
        label = r"$\rho = 0$"
    elif k == ncurves - 1:
        label = r"$\rho = 1$"
    else:
        label = None
    ax[0].plot(
        lon, I, color=cmap(k / (ncurves - 1)), label=label,
    )

# Make pretty
ax[0].set_xlim(-190, 190)
xticks = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
xticklabels = [r"  {:d}$^\circ$".format(tick) for tick in xticks]
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xticklabels)
ax[0].set_xlabel(r"$\theta$", fontsize=22)
ax[0].set_ylabel(r"intensity")
ax[0].legend(loc="lower right", fontsize=16)


# ---- BOTTOM PANEL 1 ----


# HWHM vs r
r = np.logspace(-2, 2, 300)
hwhm_fin = hwhm_empirical(ydeg, r)
hwhm_inf = hwhm(r)
ax[1].plot(
    r, hwhm_fin, "C0", label=r"$l_{\mathrm{max}} = %d$" % ydeg, zorder=2,
)
ax[1].plot(
    r, hwhm_inf, "k--", lw=1, label=r"$l_{\mathrm{max}} = \infty$", zorder=1,
)
yticks = [0, 15, 30, 45, 60, 75, 90]
yticklabels = [r"  {:d}$^\circ$".format(tick) for tick in yticks]
ax[1].set_yticks(yticks)
ax[1].set_yticklabels(yticklabels)
ax[1].set_ylim(0, 95)
ax[1].set_xlabel(r"$r$", fontsize=22)
ax[1].set_ylabel(r"$\Delta\theta$", fontsize=22)
ax[1].set_xscale("log")
ax[1].set_xlim(1e-2, 1e2)
ax[1].legend(loc="lower right", fontsize=16, framealpha=1)

# Maximum r
r_max = c[0] + c[1]
ax[1].axhline(hwhm_max, color="k", lw=1, ls="--", alpha=0.3, zorder=0)
ax[1].axvline(r_max, color="k", lw=1, ls="--", alpha=0.3, zorder=0)

# Minimum r
r_min = c[0]
ax[1].axhline(hwhm_min, color="k", lw=1, ls="--", alpha=0.3, zorder=0)
ax[1].axvline(r_min, color="k", lw=1, ls="--", alpha=0.3, zorder=0)

# Range we're modeling
ax[1].add_patch(
    patches.Rectangle(
        (r_min, hwhm_min),
        r_max - r_min,
        hwhm_max - hwhm_min,
        linewidth=1,
        edgecolor="none",
        facecolor="C0",
        alpha=0.075,
        zorder=0,
    )
)

# Plot equally spaced points in HWHM
hwhms = np.linspace(hwhm_min, hwhm_max, ncurves)
for k in range(ncurves):
    j = np.argmin(np.abs(hwhms[k] - hwhm_fin))
    ax[1].plot(
        r[j], hwhms[k], "o", ms=7, color=cmap(k / (ncurves - 1)), mec="C0"
    )


# ---- BOTTOM PANEL 2 ----


# HWHM vs rho
rho = (r - c[0]) / c[1]
ax[2].plot(
    rho, hwhm_fin, "C0", label=r"$l_{\mathrm{max}} = %d$" % ydeg, zorder=2,
)
ax[2].plot(
    rho, hwhm_inf, "k--", lw=1, label=r"$l_{\mathrm{max}} = \infty$", zorder=1,
)
yticks = [0, 15, 30, 45, 60, 75, 90]
ax[2].set_yticks(yticks)
ax[2].set_yticklabels(["" for tick in yticks])
ax[2].set_ylim(0, 95)
ax[2].set_xlabel(r"$\rho$", fontsize=22)
ax[2].set_xlim(0, 1)
ax[2].legend(loc="lower right", fontsize=16, framealpha=1)

# HWHM limits
ax[2].axhline(hwhm_max, color="k", lw=1, ls="--", alpha=0.3, zorder=0)
ax[2].axhline(hwhm_min, color="k", lw=1, ls="--", alpha=0.3, zorder=0)

# Plot equally spaced points in HWHM
hwhms = np.linspace(hwhm_min, hwhm_max, ncurves)
for k in range(ncurves):
    j = np.argmin(np.abs(hwhms[k] - hwhm_fin))
    ax[2].plot(
        rho[j], hwhms[k], "o", ms=7, color=cmap(k / (ncurves - 1)), mec="C0"
    )

# Range we're modeling
ax[2].add_patch(
    patches.Rectangle(
        (0, hwhm_min),
        1,
        hwhm_max - hwhm_min,
        linewidth=1,
        edgecolor="none",
        facecolor="C0",
        alpha=0.075,
        zorder=0,
    )
)

# Print some stats for the record
print("Delta theta range: {:.2f} - {:.2f} degrees".format(hwhm_min, hwhm_max))
print("c coeffs: {:.3f} {:.3f} {:.3f} {:.3f}".format(*c))

# Final tweaks
for axis in ax:
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
