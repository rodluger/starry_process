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
    delta = 1.0
    I = 1 - 0.5 * delta * rprime / (1 + rprime)
    for l in range(1, ydeg + 1):
        I -= 0.5 * delta * rprime * (2 + rprime) / (1 + rprime) ** (l + 1)
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


# Plot
r = np.logspace(-2, 2, 15)
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(r, hwhm(r))
ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel(r"$r'$", fontsize=16)
ax.set_ylabel(r"$\Delta\theta$ [deg]", fontsize=16)
ax.set_xscale("log")

# Maximum (max_hwhm = 75 degrees)
max_hwhm = 75.0
max_r = max_rprime(max_hwhm)
ax.axhline(max_hwhm, color="k", lw=1, ls="--", alpha=0.3)
ax.axvline(max_r, color="k", lw=1, ls="--", alpha=0.3)
ax.plot(max_r, max_hwhm, "C1o")

# Minimum (ydeg = 20, tol=1e-2)
min_r = min_rprime(20, tol=1e-2)
min_hwhm = hwhm(min_r)
ax.axhline(min_hwhm, color="k", lw=1, ls="--", alpha=0.3)
ax.axvline(min_r, color="k", lw=1, ls="--", alpha=0.3)
ax.plot(min_r, min_hwhm, "C1o")

ax.add_patch(
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

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
