import matplotlib.pyplot as plt
import numpy as np
from scipy.special import legendre as P


def BBp(ydeg=15, npts=1000, eps=1e-9, sigma=15, **kwargs):
    """
    Return the matrix product B . B+. This expands the 
    spot profile `b` in Legendre polynomials, then projects
    back onto intensity space.

    """
    theta = np.linspace(0, np.pi, npts)
    cost = np.cos(theta)
    B = np.hstack(
        [
            np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1)
            for l in range(ydeg + 1)
        ]
    )
    BInv = np.linalg.solve(B.T @ B + eps * np.eye(ydeg + 1), B.T)
    l = np.arange(ydeg + 1)
    i = l * (l + 1)
    S = np.exp(-0.5 * i / sigma ** 2)
    BInv = S[:, None] * BInv
    return B @ BInv


def b(theta, r, s=0.0033 * 180 / np.pi):
    """
    The sigmoid spot profile.

    """
    return 1 / (1 + np.exp((r - theta) / s)) - 1


def b_ylm(theta, r, s=0.0033 * 180 / np.pi, **kwargs):
    """
    The spherical harmonic expansion of the spot profile,
    projected back onto intensity space.
    
    """
    return BBp(npts=len(theta), **kwargs) @ b(theta, r, s=s)


fig, ax = plt.subplots(1)
theta = np.linspace(-90, 90, 1000)
r = [5, 10, 20, 30, 40]
cmap = plt.get_cmap("plasma")
color = [cmap(0.75 * (k / (len(r) - 1))) for k in range(len(r))]
ls = ["--"] + ["-"] * (len(r) - 1)
alpha = [0.25] + [1.0] * (len(r) - 1)
label1 = r"$\mathbf{b}(\mathbf{\vartheta})$"
label2 = r"$\mathbf{B} \mathbf{B}^+ \mathbf{b}(\mathbf{\vartheta})$"

for k in range(len(r)):
    ax.plot(
        theta,
        b_ylm(np.abs(theta), r[k]),
        ls=ls[k],
        alpha=alpha[k],
        color=color[k],
        label=r"$\rho = {}^\circ$".format(r[k]),
    )

ax.legend(loc="lower right", fontsize=12)

ax.set_xlabel(r"angle from spot center", labelpad=12)
xticks = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
ax.set_xticks(xticks)
ax.set_xticklabels([r"{}$^\circ$".format(abs(tick)) for tick in xticks])
ax.set_ylabel("intensity")

fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
