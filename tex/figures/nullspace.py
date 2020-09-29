import starry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

# Settings
ninc = 8
ntheta = 300
ydeg = 6
L = 1e9
C = 1

# Compute the posterior shrinkage & fractional rank at each inclination
map = starry.Map(ydeg, lazy=False)
map15 = starry.Map(15, lazy=False)
incs = np.linspace(20.0, 90.0, ninc)
theta = np.linspace(0, 360, ntheta, endpoint=False)
A = [None for i in range(ninc)]
A15 = [None for i in range(ninc)]
S = [None for i in range(ninc + 1)]
R15 = [None for i in range(ninc + 1)]
flux = np.random.randn(ntheta)
for i, inc in enumerate(incs):
    map.inc = inc
    A[i] = map.design_matrix(theta=theta)
    cho_C = starry.linalg.solve(
        design_matrix=A[i], data=flux, C=C, L=L, N=map.Ny,
    )[1].eval()
    S[i] = 1 - np.diag(cho_C @ cho_C.T) / L
    map15.inc = inc
    A15[i] = map15.design_matrix(theta=theta)
    R15[i] = np.linalg.matrix_rank(A15[i].T @ A15[i]) / map15.Ny

# Posterior shrinkage & fractional rank for all datasets combined
cho_C = starry.linalg.solve(
    design_matrix=np.vstack(A), data=np.tile(flux, ninc), C=C, L=L, N=map.Ny,
)[1].eval()
S[-1] = 1 - np.diag(cho_C @ cho_C.T) / L
R15[-1] = np.linalg.matrix_rank(np.vstack(A15).T @ np.vstack(A15)) / map15.Ny

# Visualize it
fig = plt.figure(figsize=(16, 8))
outer = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.2)
ax = plt.Subplot(fig, outer[-1])
rect = patches.Rectangle(
    (-0.05, -0.1),
    1.1,
    1.2,
    linewidth=1,
    edgecolor="k",
    facecolor="none",
    clip_on=False,
)
ax.add_patch(rect)
ax.axis("off")
fig.add_subplot(ax)
x = np.linspace(-1, 1, 1000)
y = np.sqrt(1 - x ** 2)
x += 0.012
y -= 0.015
for k in tqdm(range(ninc + 1)):
    inner = gridspec.GridSpecFromSubplotSpec(
        ydeg + 1, 2 * ydeg + 1, subplot_spec=outer[k], wspace=0.1, hspace=0.1
    )
    for i, l in enumerate(range(ydeg + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            map.reset()
            if k < ninc:
                map.inc = incs[k]
            if l > 0:
                map[l, m] = 1
            image = map.render()
            ax = plt.Subplot(fig, inner[(2 * ydeg + 1) * l + m + ydeg])
            ax.imshow(
                image,
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap="plasma",
                alpha=S[k][l ** 2 + l + m],
            )
            ax.plot(x, y, "k-", lw=0.5, clip_on=False)
            ax.plot(x, -y, "k-", lw=0.5, clip_on=False)
            ax.axis("off")
            fig.add_subplot(ax)

    legend = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[k], wspace=0.1, hspace=0.1
    )
    ax = plt.Subplot(fig, legend[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if k < ninc:
        ax.annotate(
            "$i = {:.0f}^\circ$".format(incs[k]),
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            va="center",
            ha="center",
            fontsize=12,
        )
    else:
        ax.annotate(
            "$\mathrm{all}\ i$",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            va="center",
            ha="center",
            fontsize=12,
        )
    ax.annotate(
        "$R = {:.2f}$".format(R15[k]),
        xy=(0.5, 0.15),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        va="center",
        ha="center",
        fontsize=12,
    )
    ax.axis("off")
    fig.add_subplot(ax)

fig.savefig("nullspace.pdf", bbox_inches="tight", dpi=300)
