import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

# Draw 1000 samples from a very long
# period sinusoid
t = np.linspace(0, 1, 300)
z = 1 + 0.1 * np.sin(
    2 * np.pi / 10 * t.reshape(1, -1)
    + 2 * np.pi * np.random.random(size=(1000, 1))
)

# Normalize each sample to the mean
znorm = z / np.mean(z, axis=1).reshape(-1, 1)

# Figure setup
fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
ax[1].set_xlabel("time", fontsize=18)
ax[0].set_ylabel("true flux", fontsize=18)
ax[1].set_ylabel("normalized flux", fontsize=18)
ax[0].margins(0, 0.1)
ax[1].margins(0, 0.1)
ax[0].set_rasterization_zorder(2)
ax[1].set_rasterization_zorder(2)

# Plot
for k in range(len(z)):
    ax[0].plot(t, z[k], color="C0", lw=1, alpha=0.1)
    ax[1].plot(t, znorm[k], color="C0", lw=1, alpha=0.1)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
