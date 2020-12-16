import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from starry_process import StarryProcess, gauss2beta
import os

# GP settings
r = 15  # spot radius in degrees
mu, sig = 30, 5  # spot latitude and std. dev. in degrees
c = 0.05  # spot contrast
n = 20  # number of spots
t = np.linspace(0, 1.5, 1000)
a, b = gauss2beta(mu, sig)

# Covariance of the original process
sp = StarryProcess(r=r, a=a, b=b, c=c, n=n, normalized=False)
Sigma = sp.cov(t).eval()

# Covariance of the normalized process
sp_norm = StarryProcess(r=r, a=a, b=b, c=c, n=n, normalized=True)
Sigma_norm = sp_norm.cov(t).eval()

# Figure setup
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
vmin = Sigma_norm.min()
vmax = Sigma_norm.max()

# Original
im = ax[0].imshow(Sigma, cmap="viridis", vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=10)

# Normalized
im = ax[1].imshow(Sigma_norm, cmap="viridis", vmin=vmin, vmax=vmax)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=10)

ax[0].set_title(r"$\mathbf{\Sigma}$", fontsize=25)
ax[1].set_title(r"$\mathbf{\tilde{\Sigma}}$", fontsize=25)

for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
