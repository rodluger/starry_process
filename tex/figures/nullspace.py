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
S = []
flux = np.random.randn(ntheta)
for i, inc in tqdm(enumerate(incs), total=len(incs)):
    map.inc = inc
    A = map.design_matrix(theta=theta)
    cho_C = starry.linalg.solve(
        design_matrix=A, data=flux, C=C, L=L, N=map.Ny,
    )[1].eval()
    S.append(1 - np.diag(cho_C @ cho_C.T) / L)

# Visualize it
fig, ax = plt.subplots(
    1, 2, figsize=(6, 4), gridspec_kw={"width_ratios": [1, 0.03]},
)
fig.subplots_adjust(wspace=0.05)

im = ax[0].imshow(
    S, vmin=0, vmax=1, aspect="auto", extent=(0, (ydeg + 1) ** 2, 0, 90),
)

cbar = plt.colorbar(im, cax=ax[1])
cbar.ax.tick_params(labelsize=8)
cbar.set_label(label="posterior shrinkage", fontsize=10)

ax[0].set_xlim(0, (ydeg ** 2))
for tick in ax[0].xaxis.get_major_ticks():
    tick.label.set_fontsize(8)
    tick.label.set_rotation(30)
for tick in ax[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ax[0].set_xticks(np.arange(2, ydeg) ** 2)
ax[0].set_xticklabels(["{}".format(l) for l in np.arange(2, ydeg)])
yticks = [0, 15, 30, 45, 60, 75, 90]
ax[0].set_yticks(yticks)
ax[0].set_yticklabels([r"{:d}$^\circ$".format(tick) for tick in yticks])
ax[0].set_ylabel("inclination", fontsize=10)
ax[0].set_xlabel("spherical harmonic degree", fontsize=10)

fig.savefig(__file__.replace("py", "pdf"), bbox_inches="tight")
