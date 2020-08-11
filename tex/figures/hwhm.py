import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import starry
from scipy.optimize import minimize
from matplotlib import patches

starry.config.quiet = True


def hwhm(rprime):
    """
    Return the half-width at half minimum as a function of r'.
    
    """
    return (
        np.arccos((2 + 3 * rprime * (2 + rprime)) / (2 * (1 + rprime) ** 3))
        * 180
        / np.pi
    )


def peak_error(ydeg, rprime):
    """
    Returns the error in the intensity at the spot center.
    
    """
    xi = 1.0
    I = 1 - 0.5 * xi * rprime / (1 + rprime)
    for l in range(1, ydeg + 1):
        I -= 0.5 * xi * rprime * (2 + rprime) / (1 + rprime) ** (l + 1)
    return np.abs(I)


def min_rprime(ydeg, tol=1e-2):
    """
    Returns the smallest value of r' such that the error on the peak
    intensity is less than `tol`.

    """
    f = lambda rprime: (peak_error(ydeg, rprime) - tol) ** 2
    res = minimize(f, 0.25)
    return res.x


def max_rprime(hmwhm_max=75):
    """
    Returns the value of r' corresponding to `hwhm_max`.
    
    """
    f = lambda rprime: (hwhm(rprime) - hmwhm_max) ** 2
    res = minimize(f, 10)
    return res.x


# Setup
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fig.subplots_adjust(wspace=0.1)
# PLOT: HWHM(rprime)
rprime = np.logspace(-2, 2, 15)
ax[0].plot(rprime, hwhm(rprime))
ax[0].set_yticks([0, 15, 30, 45, 60, 75, 90])
ax[0].set_ylim(0, 95)
ax[0].annotate(
    "'",
    xy=(0.5, 0),
    xycoords="axes fraction",
    xytext=(6, -37),
    textcoords="offset points",
    clip_on=False,
    fontsize=17,
    fontweight="bold",
    ha="center",
    va="center",
    rotation=-20,
)
ax[0].set_xlabel(r"$r$", fontsize=22)
ax[0].set_ylabel(r"$\xi\theta \, [^\circ]$", fontsize=22)
ax[0].set_xscale("log")
ax[0].set_xlim(1e-2, 1e2)

# Maximum (max_hwhm = 75 degrees)
max_hwhm = 75.0
max_r = max_rprime(max_hwhm)
ax[0].axhline(max_hwhm, color="k", lw=1, ls="--", alpha=0.3)
ax[0].axvline(max_r, color="k", lw=1, ls="--", alpha=0.3)
ax[0].plot(max_r, max_hwhm, "C2o")

# Minimum (ydeg = 20, tol=1e-2)
min_r = min_rprime(20, tol=1e-2)
min_hwhm = hwhm(min_r)
ax[0].axhline(min_hwhm, color="k", lw=1, ls="--", alpha=0.3)
ax[0].axvline(min_r, color="k", lw=1, ls="--", alpha=0.3)
ax[0].plot(min_r, min_hwhm, "C2o")

ax[0].add_patch(
    patches.Rectangle(
        (min_r, min_hwhm),
        max_r - min_r,
        max_hwhm - min_hwhm,
        linewidth=1,
        edgecolor="none",
        facecolor="C0",
        alpha=0.15,
    )
)


logrprime = lambda r: np.log10(min_r + (max_r - min_r) * r)

axt = ax[0].twiny()
axt.set_xlim(-2, 2)
ticks = logrprime(np.array([0, 1]))
ticklabels = ["0", "1"]
minorticks = logrprime(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
axt.set_xticks(minorticks, minor=True)
axt.set_xticks(ticks)
axt.set_xticklabels(ticklabels)
axt.set_xlabel(r"$r$", fontsize=22)


# PLOT: Minimum HWHM(ydeg)
l = np.arange(0, 31, dtype=int)
for tol, style in zip([1e-1, 1e-2, 1e-3], ["C0--", "C0-", "C0:"]):
    r = hwhm(np.array([min_rprime(ydeg, tol=tol) for ydeg in l]))
    ax[1].plot(l, r, style, label=tol, zorder=-1)
legend = ax[1].legend(loc="upper right", fontsize=10, title=r"tolerance", ncol=3)
plt.setp(legend.get_title(), fontsize=11)
ax[1].axhline(75, color="k", lw=1, ls="--", alpha=0.3)
ax[1].axhline(min_hwhm, color="k", lw=1, ls="--", alpha=0.3)

ax[1].set_ylim(0, 95)
ax[1].set_xlim(0, 30)
ax[1].set_yticks([0, 15, 30, 45, 60, 75, 90])
ax[1].set_xlabel(r"$l_{\mathrm{max}}$", fontsize=22)
ax[1].set_ylabel(r"$\xi\theta_{\mathrm{min}} \, [^\circ]$", fontsize=22)
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].plot(20, min_hwhm, "C2o")

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
