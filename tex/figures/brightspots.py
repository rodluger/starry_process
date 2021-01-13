import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from starry_process import StarryProcess
import os
from tqdm import tqdm

nsamples = 5
cmap = plt.get_cmap("plasma_r")
color = [cmap(0.75), cmap(0.25)]
label = ["dark", "bright"]
t = np.linspace(0, 1, 1000)
kwargs = dict(r=10, a=0.40, b=0.27, N=10, seed=0)
inc = 75.0

fig, ax = plt.subplots(
    3,
    nsamples + 1,
    figsize=(12, 5),
    gridspec_kw={
        "height_ratios": [1, 1, 1],
        "width_ratios": np.append(np.ones(nsamples), 0.1),
    },
)


for n, c in enumerate([0.1, -0.1]):

    # Draw samples
    sp = StarryProcess(marginalize_over_inclination=False, c=c, **kwargs)
    y = sp.sample_ylm(nsamples=nsamples).eval()
    flux = 1e3 * sp.flux(y, t, i=inc).eval()

    for k in range(nsamples):
        sp.visualize(y[k], ax=ax[n, k], vmin=0.8, vmax=1.2)
        ax[n, k].set_ylim(-1.5, 2.25)
        ax[n, k].set_rasterization_zorder(1)
        ax[2, k].plot(t, flux[k], color=color[n], lw=0.75)

        if k == 0:
            ax[2, k].spines["top"].set_visible(False)
            ax[2, k].spines["right"].set_visible(False)
            ax[2, k].set_xlabel("rotations", fontsize=8)
            ax[2, k].set_ylabel("flux [ppt]", fontsize=8)
            ax[2, k].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            for tick in (
                ax[2, k].xaxis.get_major_ticks()
                + ax[2, k].yaxis.get_major_ticks()
            ):
                tick.label.set_fontsize(6)
            ax[2, k].tick_params(direction="in")
        else:
            ax[2, k].axis("off")

    cax = inset_axes(ax[n, -1], width="70%", height="50%", loc="lower center")
    cbar = fig.colorbar(ax[n, k].images[0], cax=cax, orientation="vertical")
    cbar.set_label("intensity", fontsize=8)
    cbar.set_ticks([0.8, 1, 1.2])
    cbar.ax.tick_params(labelsize=6)
    ax[n, -1].axis("off")

lax = inset_axes(ax[2, -1], width="80%", height="100%", loc="center right")
for n in range(2):
    lax.plot(0, 0, color=color[n], lw=1, label=label[n])
lax.legend(loc="center left", fontsize=8, frameon=False)
lax.axis("off")
ax[2, -1].axis("off")

dy = max([max(np.abs(ax[2, k].get_ylim())) for k in range(nsamples)])
for k in range(nsamples):
    ax[2, 0].set_ylim(-dy, dy)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
