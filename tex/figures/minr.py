import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import starry
from scipy.optimize import minimize

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


# Plot
fig, ax = plt.subplots(1, figsize=(6, 4))
l = np.arange(0, 31, dtype=int)
for tol, style in zip([1e-1, 1e-2, 1e-3], ["C0--", "C0-", "C0:"]):
    r = hwhm(np.array([min_rprime(ydeg, tol=tol) for ydeg in l]))
    plt.plot(l, r, style, label=tol, zorder=-1)
legend = ax.legend(loc="upper right", fontsize=10, title=r"tolerance", ncol=3)
plt.setp(legend.get_title(), fontsize=11)
ax.axhline(75, color="k", ls="--")
ax.axhspan(75, 100, color="r", alpha=0.25)
ax.set_ylim(0, 95)
ax.set_xlim(0, 30)
ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
ax.set_xlabel("spherical harmonic degree", fontsize=16)
ax.set_ylabel(r"minimum $\Delta\theta$ [deg]", fontsize=16)

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
