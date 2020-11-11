import starry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

# Settings
ninc = 50
ntheta = 300
ydeg = 14
L = 1e9
C = 1

# Compute the posterior shrinkage & fractional rank at each inclination
map = starry.Map(ydeg, lazy=False)
incs = np.linspace(0.0, 90.0, ninc)[::-1]
theta = np.linspace(0, 360, ntheta, endpoint=False)
A = []
S = []
Sc = []
R = []
Rc = []
flux = np.random.randn(ntheta)
for i, inc in tqdm(enumerate(incs), total=len(incs)):

    # At this inclination
    map.inc = inc
    A.append(map.design_matrix(theta=theta))
    cho_C = starry.linalg.solve(
        design_matrix=A[-1], data=flux, C=C, L=L, N=map.Ny,
    )[1].eval()
    S.append(1 - np.diag(cho_C @ cho_C.T) / L)

    # Cumulative
    cho_C = starry.linalg.solve(
        design_matrix=np.vstack(A),
        data=np.tile(flux, i + 1),
        C=C,
        L=L,
        N=map.Ny,
    )[1].eval()
    Sc.append(1 - np.diag(cho_C @ cho_C.T) / L)

    # np.linalg.matrix_rank(np.vstack(A15).T @ np.vstack(A15)) / map15.Ny

# Visualize it
fig, ax = plt.subplots(
    1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1, 0.05]},
)
fig.subplots_adjust(wspace=0.1)

ax[0].imshow(
    S, vmin=0, vmax=1, aspect="auto", extent=(0, (ydeg + 1) ** 2, 0, 90),
)
im = ax[1].imshow(
    Sc, vmin=0, vmax=1, aspect="auto", extent=(0, (ydeg + 1) ** 2, 0, 90),
)
cbar = plt.colorbar(im, cax=ax[2], label="posterior shrinkage")
cbar.ax.tick_params(labelsize=10)

for axis in [ax[0], ax[1]]:
    axis.set_yticks([0, 15, 30, 45, 60, 75, 90])
    axis.set_xticks(np.arange(2, ydeg + 1) ** 2)
    axis.set_xticklabels(["{}".format(l) for l in np.arange(2, ydeg + 1)])
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        tick.label.set_rotation(30)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    axis.set_xlabel("spherical harmonic degree")
ax[1].set_yticklabels([])
ax[0].set_ylabel("inclination")

ax[0].set_title("fixed")
ax[1].set_title("cumulative")

fig.savefig("nullspace.pdf", bbox_inches="tight", dpi=300)
