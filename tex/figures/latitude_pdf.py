import numpy as np
import matplotlib.pyplot as plt
from starry_process import StarryProcess, gauss2beta, beta2gauss
import theano
import theano.tensor as tt
from scipy.stats import norm
import os

phi = np.linspace(-90, 90, 1000)
a = tt.dscalar()
b = tt.dscalar()
pdf = theano.function([a, b], StarryProcess(a=a, b=b).latitude.pdf(phi))

pdf_gauss = lambda mu, sig: 0.5 * (
    norm.pdf(phi, mu, sig) + norm.pdf(phi, -mu, sig)
)

sig = [1, 5, 10, 40]
mu = [0, 30, 60, 85]

fig, ax = plt.subplots(len(mu), len(sig), figsize=(12, 8), sharex=True)


for i in range(len(sig)):
    for j in range(len(mu)):
        ax[i, j].plot(phi, pdf(*gauss2beta(mu[j], sig[i])), "C0-")

        if mu[j] == 0 and sig[i] == 40:
            ax[i, j].plot(
                phi, 0.5 * np.pi / 180 * np.cos(phi * np.pi / 180), "C1--"
            )
            ax[i, j].set_facecolor("#1f77b422")
        else:
            ax[i, j].plot(phi, pdf_gauss(mu[j], sig[i]), "C1--", lw=1)

        ax[i, j].annotate(
            r"$a = {:.2f}$".format(gauss2beta(mu[j], sig[i])[0])
            + "\n"
            + r"$b = {:.2f}$".format(gauss2beta(mu[j], sig[i])[1]),
            xy=(0, 1),
            xytext=(5, -5),
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=7,
        )

for i in range(len(sig)):
    ax[i, 0].get_shared_y_axes().join(*ax[i])
    for j in range(1, len(mu)):
        ax[i, j].set_yticklabels([])
    ax[i, 0].set_ylabel("probability density", fontsize=8)
    for tick in ax[i, 0].yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax[i, 0].annotate(
        r"$\sigma_\phi = {}^\circ$".format(sig[i]),
        xy=(0, 0.5),
        xytext=(-55, 0),
        xycoords="axes fraction",
        textcoords="offset points",
        va="center",
        ha="right",
        fontsize=18,
        clip_on=False,
        rotation=90,
    )

for j in range(len(mu)):
    xticks = [-90, -60, -30, 0, 30, 60, 90]
    ax[-1, j].set_xticks(xticks)
    ax[-1, j].set_xticklabels(
        [r"{}$^\circ$".format(abs(tick)) for tick in xticks]
    )
    ax[-1, j].set_xlabel("latitude", fontsize=10)
    for tick in ax[-1, j].xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    ax[0, j].set_title(
        r"$\mu_\phi = {}^\circ$".format(mu[j]), fontsize=18, y=1.05
    )

for axis in ax.flatten():
    axis.margins(0, None)

fig.align_ylabels(ax[:, 0])

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
