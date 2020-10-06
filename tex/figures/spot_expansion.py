import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
from scipy.optimize import minimize
from scipy.special import legendre
import logging

logger = logging.getLogger("spot_expansion")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def hwhm(rhop):
    """
    Return the theoretical half-witdth at half minimum as a function of rho'.
    
    """
    return (
        np.arccos((2 + 3 * rhop * (2 + rhop)) / (2 * (1 + rhop) ** 3))
        * 180
        / np.pi
    )


def hwhm_inv(hwhm):
    """
    Return rho' as a function of the theoretical half-width at half minimum.
    
    """
    theta = hwhm * np.pi / 180
    return (1 + np.cos(2 * theta / 3) + np.sqrt(3) * np.sin(2 * theta / 3)) / (
        2 * np.cos(theta)
    ) - 1


def rhop_max(hwhm_max=60):
    """
    Returns the value of rho' corresponding to `hwhm_max`.
    
    """
    f = lambda rhop: (hwhm(rhop) - hwhm_max) ** 2
    res = minimize(f, 2.0)
    return res.x[0]


def corr(rhop, c):
    """Intensity correction function."""
    rho = (rhop - c[0]) / c[1]
    return 1 + c[2] * (1 - rho) ** c[3]


def I(ydeg, rhop, theta, c=None):
    """
    Return the intensity at polar angle `theta` (in deg) away from
    the center of a spot of radius rho' expanded to degree `ydeg`.
    
    """
    # Compute the Legendre expansion
    cost = np.cos(theta * np.pi / 180)
    term = np.sum(
        [(1 + rhop) ** -l * legendre(l)(cost) for l in range(ydeg + 1)], axis=0
    )
    I = 0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # Apply the intensity correction
    if c is not None:
        I *= corr(rhop, c)

    return I


def get_c(ydeg, hwhm_max=75, hwhm_min=15, npts=500):
    """
    Return the coefficients for the radius transformation.

    """
    c = np.zeros(4)

    # Minimum r: we need to optimize numerically
    loss = lambda p: (hwhm_empirical(ydeg, p[0]) - hwhm_min) ** 2
    res = minimize(loss, hwhm_inv(hwhm_min))
    rhopmin = res.x[0]
    c[0] = rhopmin

    # Maximum r (easy)
    rhopmax = rhop_max(hwhm_max=hwhm_max)
    c[1] = rhopmax - rhopmin

    # Now compute the coefficients of the intensity
    # correction, c[2] and c[3].

    # Array over which to compute the loss
    rhop = np.linspace(rhopmin + 1e-6, rhopmax - 1e-6, npts)

    # Get the actual (absolute value of the) intensity at the peak
    l = np.arange(ydeg + 1).reshape(1, -1)
    term = np.sum((1 + rhop.reshape(-1, 1)) ** -l, axis=-1)
    I = -0.5 * rhop * (1 - (2 + rhop) / (1 + rhop) * term)

    # This is the factor by which we need to normalize the function
    norm = 1.0 / I

    # Find the coefficients of the fit (least squares)
    diff = lambda p: np.sum((norm - corr(rhop, [c[0], c[1], p[0], p[1]])) ** 2)
    res = minimize(diff, [0.1, 50.0])
    c[2:] = res.x

    # Log the error info
    logger.info(
        "Delta theta range: {:.2f} - {:.2f} degrees".format(hwhm_min, hwhm_max)
    )
    logger.info("c coeffs: {:.3f} {:.3f} {:.3f} {:.3f}".format(*c))
    logger.info(
        "Maximum intensity |error|: {:.2e}".format(
            np.max(np.abs(norm - corr(rhop, c)))
        )
    )
    logger.info(
        "Average intensity |error|: {:.2e}".format(
            np.mean(np.abs(norm - corr(rhop, c)))
        )
    )

    return c


def hwhm_empirical(ydeg, rhop):
    """
    Return the empirical half-width at half minimum as a function of rho'.
    
    """
    # Setup
    rhop = np.atleast_1d(rhop)
    hwhm_empirical = np.zeros_like(rhop)

    # Find the HWHM numerically for each radius
    for k in range(len(rhop)):

        halfmax = 0.5 * I(ydeg, rhop[k], 0)

        def loss(theta):
            return (I(ydeg, rhop[k], theta) - halfmax) ** 2

        res = minimize(loss, hwhm(max(0.1, rhop[k])))
        hwhm_empirical[k] = res.x[0]

    return hwhm_empirical


# Settings
ydeg = 15
hwhm_min = 15
hwhm_max = 75
ncurves = 10
cmap = lambda x: plt.get_cmap("plasma")(0.85 * (1 - x))

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


# Longitude array
lon = np.linspace(-180, 180, 1000)

# We need to find the values of r corresponding
# to `ncurves` equally spaced values of HWHM
# We'll do nearest-neighbor, since it's fastest
hwhms = np.linspace(hwhm_min, hwhm_max, ncurves)
rhop_ = np.logspace(-2, 1, 100)
h_ = hwhm_empirical(ydeg, rhop_)

# Plot the intensity profile for each radius
for k in range(ncurves):

    # The current value of r
    rhop = rhop_[np.argmin(np.abs(hwhms[k] - h_))]

    # Plot it
    if k == 0:
        label = r"$\rho = 0$"
    elif k == ncurves - 1:
        label = r"$\rho = 1$"
    else:
        label = None

    ax[0].plot(
        lon, I(ydeg, rhop, lon, c), color=cmap(k / (ncurves - 1)), label=label,
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


# HWHM vs rho'
rhop = np.logspace(-2, 2, 300)
hwhm_fin = hwhm_empirical(ydeg, rhop)
hwhm_inf = hwhm(rhop)
ax[1].plot(
    rhop, hwhm_fin, "C0", label=r"$l_{\mathrm{max}} = %d$" % ydeg, zorder=2,
)
ax[1].plot(
    rhop,
    hwhm_inf,
    "k--",
    lw=1,
    label=r"$l_{\mathrm{max}} = \infty$",
    zorder=1,
)
yticks = [0, 15, 30, 45, 60, 75, 90]
yticklabels = [r"  {:d}$^\circ$".format(tick) for tick in yticks]
ax[1].set_yticks(yticks)
ax[1].set_yticklabels(yticklabels)
ax[1].set_ylim(0, 95)
ax[1].set_xlabel(r"$\rho\prime$", fontsize=22)
ax[1].set_ylabel(r"$\Delta\theta$", fontsize=22)
ax[1].set_xscale("log")
ax[1].set_xlim(1e-2, 1e2)
ax[1].legend(loc="lower right", fontsize=16, framealpha=1)

# Maximum r
rhopmax = c[0] + c[1]
ax[1].axhline(hwhm_max, color="k", lw=1, ls="--", alpha=0.3, zorder=0)
ax[1].axvline(rhopmax, color="k", lw=1, ls="--", alpha=0.3, zorder=0)

# Minimum r
rhopmin = c[0]
ax[1].axhline(hwhm_min, color="k", lw=1, ls="--", alpha=0.3, zorder=0)
ax[1].axvline(rhopmin, color="k", lw=1, ls="--", alpha=0.3, zorder=0)

# Range we're modeling
ax[1].add_patch(
    patches.Rectangle(
        (rhopmin, hwhm_min),
        rhopmax - rhopmin,
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
        rhop[j], hwhms[k], "o", ms=7, color=cmap(k / (ncurves - 1)), mec="C0"
    )


# ---- BOTTOM PANEL 2 ----


# HWHM vs rho
rho = (rhop - c[0]) / c[1]
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

# Final tweaks
for axis in ax:
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
