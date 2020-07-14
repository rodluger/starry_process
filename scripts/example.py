from starry_process import SP
import numpy as np
import matplotlib.pyplot as plt
import starry

plt.style.use("default")
starry.config.lazy = False


# Degree of the expansion
ydeg = 15

# Parameters of the beta distribution in cos(lat)
# This combination gives an *approximately*
# uniform distribution of spots over the sphere
alpha = 2
beta = 0.5

# Log spot size distribution
ln_sigma_mu = -5
ln_sigma_sigma = 0.01

# Log spot amplitude distribution
ln_amp_mu = -2.3
ln_amp_sigma = 0.1

# Sign of the spot amplitude (1 = positive)
sign = 1

# Instantiate the starry process

print("1")

sp = SP(
    ydeg,
    alpha=alpha,
    beta=beta,
    ln_sigma_mu=ln_sigma_mu,
    ln_sigma_sigma=ln_sigma_sigma,
    ln_amp_mu=ln_amp_mu,
    ln_amp_sigma=ln_amp_sigma,
    sign=sign,
)

# Get the mean and covariance
# of the starry process
mu = sp.mu_y
cov = sp.cov_y

print("2")

# Plot them
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
n = np.arange(len(mu))
ax[0].plot(n, mu)
ax[0].set_yscale("log")
z = np.log10(np.abs(sp.cov_y))
z[np.isnan(z)] = -10
z[z < -10] = -10
img = ax[1].imshow(z, vmin=-9)
plt.colorbar(img, ax=ax[1])

# Draw 25 maps from the prior
samples = sp.draw_y(ndraws=25)

# Plot them
fig, ax = plt.subplots(5, 5, figsize=(15, 7))
ax = ax.flatten()
map = starry.Map(ydeg=ydeg)
for i, axis in enumerate(ax):
    map[1:, :] = samples[i, 1:]
    img = map.render(projection="moll")
    axis.imshow(img, origin="lower", cmap="plasma", extent=(-2, 2, -1, 1))
    axis.axis("off")
plt.show()
