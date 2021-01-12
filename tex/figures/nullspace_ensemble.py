import starry
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Settings
ntheta = 50
ydeg = 15
ydeg_pad = 3
L = 1e9
C = 1
ninc = [1, 3, 10, 30]
kpn = 300
clobber = False

# Compute
DATA_FILE = (
    os.path.abspath(__file__)
    .replace("figures", "figures/data")
    .replace(".py", ".npz")
)
if clobber or not os.path.exists(DATA_FILE):
    map = starry.Map(ydeg + ydeg_pad, lazy=False)
    theta = np.linspace(0, 360, ntheta, endpoint=False)
    S = np.empty((len(ninc), kpn, map.Ny))
    np.random.seed(0)
    for n in tqdm(
        range(len(ninc)), disable=bool(int(os.getenv("NOTQDM", "0")))
    ):
        for k in range(kpn):
            A = np.empty((0, map.Ny))
            for _ in range(ninc[n]):
                map.inc = 180 / np.pi * np.arccos(np.random.uniform(0, 1))
                A = np.vstack((A, map.design_matrix(theta=theta)))
            cho_C = starry.linalg.solve(
                design_matrix=A,
                data=np.random.randn(A.shape[0]),
                C=C,
                L=L,
                N=map.Ny,
            )[1].eval()
            S[n, k] = 1 - np.diag(cho_C @ cho_C.T) / L
    S = np.array(S, dtype="float32")
    np.savez(DATA_FILE, S=S)
else:
    S = np.load(DATA_FILE)["S"]

# Plot
fig, ax = plt.subplots(1)
for n in tqdm(range(len(ninc)), disable=bool(int(os.getenv("NOTQDM", "0")))):
    for k in range(kpn):
        ax.plot(S[n, k], color="C{}".format(n), lw=0.75, alpha=0.01, zorder=-1)
    ax.plot(
        np.mean(S[n], axis=0),
        color="C{}".format(n),
        lw=1,
        label=ninc[n],
        zorder=1,
    )
ax.set_rasterization_zorder(0)
ax.set_xlim(0, (ydeg + 1) ** 2 - 1)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(10)
    tick.label.set_rotation(30)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
l = np.arange(2, ydeg + 1)
ax.set_xticks(l ** 2 + l)
ax.set_xticklabels(["{}".format(l) for l in np.arange(2, ydeg + 1)])
ax.set_xlabel("spherical harmonic degree")
ax.set_ylabel("posterior shrinkage")
leg = ax.legend(loc="upper right", title="orientations", fontsize=10)
leg.get_title().set_fontsize(8)
leg.get_title().set_fontweight("bold")

# Print some info
print(
    "Total shrinkage for each scenario:",
    np.mean(S[:, :, : (ydeg + 1) ** 2], axis=(1, 2)),
)

# Save
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
