import starry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

# Settings
ninc = 50
ntheta = 300
ydeg = 16
L = 1e9
C = 1

# Compute the posterior shrinkage & fractional rank at each inclination
map = starry.Map(ydeg, lazy=False)
incs = np.linspace(0.0, 90.0, ninc)[::-1]
theta = np.linspace(0, 360, ntheta, endpoint=False)
A = []
S = []
flux = np.random.randn(ntheta)
for i, inc in tqdm(enumerate(incs), total=len(incs)):
    map.inc = inc
    A.append(map.design_matrix(theta=theta))
    cho_C = starry.linalg.solve(
        design_matrix=A[-1], data=flux, C=C, L=L, N=map.Ny,
    )[1].eval()
    S.append(1 - np.diag(cho_C @ cho_C.T) / L)

# Cumulative shrinkage
cho_C = starry.linalg.solve(
    design_matrix=np.vstack(A), data=np.tile(flux, i + 1), C=C, L=L, N=map.Ny,
)[1].eval()
Sc = 1 - np.diag(cho_C @ cho_C.T) / L

# Visualize it
fig, ax = plt.subplots(
    2,
    2,
    figsize=(6, 4),
    gridspec_kw={"height_ratios": [0.05, 1], "width_ratios": [1, 0.03]},
)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

ax[0, 0].imshow(
    Sc.reshape(1, -1),
    vmin=0,
    vmax=1,
    aspect="auto",
    extent=(0, (ydeg + 1) ** 2, 0, 1),
)
ax[0, 0].annotate(
    "all",
    xy=(0, 0.5),
    xytext=(-10, 0),
    ha="right",
    va="bottom",
    clip_on=False,
    fontsize=10,
)
im = ax[1, 0].imshow(
    S, vmin=0, vmax=1, aspect="auto", extent=(0, (ydeg + 1) ** 2, 0, 90),
)

cbar = plt.colorbar(im, cax=ax[1, 1], label="posterior shrinkage")
cbar.ax.tick_params(labelsize=8)

for axis in [ax[0, 0], ax[1, 0]]:
    axis.set_xlim(0, (ydeg ** 2))
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_rotation(30)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)

ax[0, 1].axis("off")
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[1, 0].set_xticks(np.arange(2, ydeg) ** 2)
ax[1, 0].set_xticklabels(["{}".format(l) for l in np.arange(2, ydeg)])
yticks = [0, 15, 30, 45, 60, 75, 90]
ax[1, 0].set_yticks(yticks)
ax[1, 0].set_yticklabels([r"{:d}$^\circ$".format(tick) for tick in yticks])
ax[1, 0].set_ylabel("inclination")
ax[1, 0].set_xlabel("spherical harmonic degree")

fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
