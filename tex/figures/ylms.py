import starry
import matplotlib.pyplot as plt
import numpy as np
import os

# Settings
lmax = 5
res = 300

# Set up the plot
fig, ax = plt.subplots(lmax + 1, 2 * lmax + 1, figsize=(9, 6))
fig.subplots_adjust(hspace=0)
for axis in ax.flatten():
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
for l in range(lmax + 1):
    ax[l, lmax - l].set_ylabel(
        r"$l = %d$" % l,
        rotation="horizontal",
        labelpad=15,
        y=0.35,
        fontsize=11,
        alpha=0.5,
    )
for j, m in enumerate(range(-lmax, lmax + 1)):
    if m < 0:
        ax[-1, j].set_xlabel(
            r"$m {=} $-$%d$" % -m,
            labelpad=5,
            fontsize=11,
            rotation="45",
            x=0.3,
            alpha=0.5,
        )
    else:
        ax[-1, j].set_xlabel(
            r"$m = %d$" % m,
            labelpad=5,
            fontsize=11,
            rotation=45,
            x=0.35,
            alpha=0.5,
        )

# Loop over the orders and degrees
map = starry.Map(lmax, lazy=False)
for i, l in enumerate(range(lmax + 1)):
    for j, m in enumerate(range(-l, l + 1)):

        # Offset the index for centered plotting
        j += lmax - l

        # Compute the spherical harmonic
        map.reset()
        if l > 0:
            map[l, m] = 1.0
        map.show(ax=ax[i, j], grid=False)
        ax[i, j].axis("on")
        ax[i, j].set_rasterization_zorder(1)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
