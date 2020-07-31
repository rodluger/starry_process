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


def max_rprime(hmwhm_max=75):
    """
    Returns the value of r' corresponding to `hwhm_max`.
    
    """
    f = lambda rprime: (hwhm(rprime) - hmwhm_max) ** 2
    res = minimize(f, 10)
    return res.x


def get_c(ydeg, tol=1e-2, hwhm_max=75):
    """
    Returns the c_0, c_1 coefficients for the radius.

    """
    rmin = min_rprime(ydeg, tol=tol)
    rmax = max_rprime(hmwhm_max=hwhm_max)
    c0 = rmin
    c1 = rmax - rmin
    return c0, c1


ydeg = 20
map = starry.Map(ydeg, lazy=False)

# Normalized radius in (0, 1]
r = np.logspace(-2, 1, 15)

# Convert to actual radius used in the expansion, r'
c0, c1 = get_c(ydeg)
rprime = c0 + c1 * r

# Add a few values below the minimum to show the ringing
rprime = np.append([c0 / 8, c0 / 4, c0 / 2], rprime)

# Spot contrast
delta = 1

# Longitude array
lon = np.linspace(-180, 180, 1000)

# Set up the plot
fig, ax = plt.subplots(1, figsize=(12, 4))
cmap = plt.get_cmap("viridis")
l = np.arange(1, map.ydeg + 1)

# Plot the intensity profile for each radius
for k in range(len(r)):

    # Legendre expansion
    x = np.zeros(map.Ny)
    x[0] = 1 - 0.5 * delta * rprime[k] * (1 + rprime[k]) ** -1
    x[l * (l + 1)] = (
        -delta
        / np.sqrt(2 * l + 1)
        * (
            (1 + rprime[k]) ** -(l + 1) * rprime[k]
            + 0.5 * (1 + rprime[k]) ** -(l + 1) * rprime[k] ** 2
        )
    )
    map[:, :] = x

    if k < 3:
        # These are the profiles with r' that's too small
        # They all have ringing!
        if k == 0:
            label = r"$r < 0$"
        else:
            label = None
        ax.plot(
            lon[100:-100],
            np.pi * map.intensity(lon=lon[100:-100]),
            "k--",
            lw=0.5,
            alpha=0.5,
            label=label,
        )
    else:
        # These are good, in the range 0 < r <= 1
        if k == 4:
            label = r"$r = 0$"
        elif k == len(r) - 1:
            label = r"$r = 1$"
        else:
            label = None
        ax.plot(
            lon,
            np.pi * map.intensity(lon=lon),
            color=cmap(k / (len(r) - 1)),
            label=label,
        )

# Make pretty
ax.set_xlim(-190, 190)
ax.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
ax.set_xlabel(r"$\theta$ [deg]")
ax.set_ylabel(r"intensity")
ax.legend(loc="lower right")

# We're done
fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
