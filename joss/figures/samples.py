import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from starry_process import StarryProcess
import starry
import os
from tqdm import tqdm

nsamples = 5
norm = Normalize(vmin=0.5, vmax=1.1)
incs = [15, 30, 45, 60, 75, 90]
t = np.linspace(0, 4, 1000)
cmap = plt.get_cmap("plasma_r")
color = lambda i: cmap(0.1 + 0.8 * i / (len(incs) - 1))
kwargs = [
    dict(r=10, a=0.40, b=0.27, c=0.15, n=10, seed=7),
    dict(r=15, a=0.31, b=0.36, c=0.075, n=10, seed=8),
]

map = starry.Map(15, lazy=False)

fig, ax = plt.subplots(
    2 * len(kwargs),
    nsamples + 1,
    figsize=(12, 2.5 * len(kwargs)),
    gridspec_kw={
        "height_ratios": np.tile([1, 0.5], len(kwargs)),
        "width_ratios": np.append(np.ones(nsamples), 0.1),
    },
)
ax = np.swapaxes(
    np.swapaxes(ax.T.reshape(nsamples + 1, len(kwargs), 2), 0, 2), 0, 1
)


for n in range(len(kwargs)):

    # Draw samples
    sp = StarryProcess(marginalize_over_inclination=False, **kwargs[n])
    y = sp.sample_ylm(nsamples=nsamples).eval()

    # Normalize so that the background photosphere
    # has unit intensity (for plotting)
    y[:, 0] += 1
    y *= np.pi

    for k in range(nsamples):
        map[:, :] = y[k]
        map.show(ax=ax[n, 0, k], projection="moll", norm=norm)
        ax[n, 0, k].set_ylim(-1.5, 2.25)
        ax[n, 0, k].set_rasterization_zorder(1)
        for i, inc in enumerate(incs):
            map.inc = inc
            flux = map.flux(theta=360.0 * t)
            flux -= np.mean(flux)
            flux *= 1e3
            ax[n, 1, k].plot(t, flux, color=color(i), lw=0.75)

        if k == 0:
            ax[n, 1, k].spines["top"].set_visible(False)
            ax[n, 1, k].spines["right"].set_visible(False)
            ax[n, 1, k].set_xlabel("rotations", fontsize=8)
            ax[n, 1, k].set_ylabel("flux [ppt]", fontsize=8)
            ax[n, 1, k].set_xticks([0, 1, 2, 3, 4])
            for tick in (
                ax[n, 1, k].xaxis.get_major_ticks()
                + ax[n, 1, k].yaxis.get_major_ticks()
            ):
                tick.label.set_fontsize(6)
            ax[n, 1, k].tick_params(direction="in")
        else:
            ax[n, 1, k].axis("off")

    ax[n, 0, 0].annotate(
        ["(a)", "(b)", "(c)", "(d)"][n],
        xy=(0, 0.5),
        xytext=(-10, -7),
        xycoords="axes fraction",
        textcoords="offset points",
        fontsize=12,
        ha="right",
        va="center",
        clip_on=False,
    )

    cax = inset_axes(
        ax[n, 0, -1], width="70%", height="50%", loc="lower center"
    )
    cbar = fig.colorbar(ax[n, 0, k].images[0], cax=cax, orientation="vertical")
    cbar.set_label("intensity", fontsize=8)
    cbar.set_ticks([0.5, 0.75, 1])
    cbar.ax.tick_params(labelsize=6)
    ax[n, 0, -1].axis("off")

    lax = inset_axes(
        ax[n, 1, -1], width="80%", height="100%", loc="center right"
    )
    for i, inc in enumerate(incs):
        lax.plot(0, 0, color=color(i), lw=1, label=r"{}$^\circ$".format(inc))
    lax.legend(loc="center left", fontsize=5, frameon=False)
    lax.axis("off")
    ax[n, 1, -1].axis("off")

    dy = max([max(np.abs(ax[n, 1, k].get_ylim())) for k in range(nsamples)])
    for k in range(nsamples):
        ax[n, 1, 0].set_ylim(-dy, dy)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
