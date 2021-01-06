import numpy as np
import matplotlib.pyplot as plt
from starry_process import StarryProcess
import starry
from scipy.linalg import cho_factor
from matplotlib.colors import Normalize
import os

# GP settings
ydeg = 15
tau = 20.0
p = 1.0
tmax = 10.0
npts = 1000
inc = 60.0
kwargs = dict(r=15)
seed = 0

# Plotting settings
nimg = 5
pad = 85
vmin = -0.15
vmax = 0.075

# Spatial covariance
np.random.seed(seed)
sp = StarryProcess(ydeg=ydeg, seed=seed, **kwargs)
cov_y = sp.cov_ylm.eval()
Ly = np.tril(cho_factor(cov_y, lower=True)[0])
Ny = Ly.shape[0]

# Temporal covariance
t = np.linspace(0, tmax, npts)
kernel = lambda t1, t2: np.exp(-((t1 - t2) ** 2) / (2 * tau))
Nt = len(t)
eps = 1e-12
cov_t = kernel(t.reshape(1, -1), t.reshape(-1, 1))
Lt = np.tril(cho_factor(cov_t + eps * np.eye(Nt), lower=True)[0])

# Sample the map and the flux
# These operations are identical to the extremely memory
# intensive, excrutiatingly slow operations
#
#     L = np.kron(Ly, Lt)
#     y = (L @ U.reshape(-1)).reshape(Ny, Nt).T
#     flux = np.diag(A @ y.T)
#
map = starry.Map(ydeg, inc=inc, lazy=False)
A = map.design_matrix(theta=360 / p * t)
y = np.zeros((Nt, Ny))
flux = np.zeros(Nt)
U = np.random.randn(Ny, Nt)
for j in range(Ny):
    flux += (A @ Ly.T[j]) * (Lt @ U[j])
    for k in range(Nt):
        y[k] += Ly.T[j] * (Lt[k] @ U[j])

# Render the images
idx = np.array(np.linspace(pad, Nt - 1 - pad, nimg, endpoint=True), dtype=int)
image = [None for k in range(nimg)]
for k in range(nimg):
    map[:, :] = y[idx[k]]
    image[k] = map.render(projection="moll")

# Plot everything
fig = plt.figure(figsize=(12, 6))
axtop = [plt.subplot2grid((5, nimg), (0, k)) for k in range(nimg)]
axbot = plt.subplot2grid((5, nimg), (1, 0), colspan=nimg, rowspan=4)
norm = Normalize(vmin=vmin, vmax=vmax)
for k in range(nimg):
    map.show(ax=axtop[k], image=image[k], projection="moll", norm=norm)
axbot.plot(t, 1e3 * (flux - flux.mean()))
for k in range(nimg):
    plt.axvline(t[idx[k]], color="k", ls="--", lw=1, alpha=0.5)
axbot.set_xlabel("time [days]", fontsize=18)
axbot.set_ylabel("flux [ppt]", fontsize=18)
axbot.margins(0, 0.2)

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
