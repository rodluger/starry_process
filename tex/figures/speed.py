import numpy as np
import matplotlib.pyplot as plt
from starry_process import StarryProcess
import theano
import theano.tensor as tt
from tqdm import tqdm
import os
import time


# Compile the function to compute the likelihood
def get_loglike_function(ydeg, marginalize_over_inclination=False):
    t = tt.dvector()
    return theano.function(
        [t],
        StarryProcess(
            ydeg=ydeg,
            marginalize_over_inclination=marginalize_over_inclination,
        ).log_likelihood(t, tt.ones_like(t), 1.0),
    )


# Settings
ydeg = 15
npts = np.sort(list(set(list(np.array(np.logspace(0, 4, 20), dtype=int)))))
ncalls = 10

# Compute
tloglike = np.zeros((2, len(npts)))
for i, marginalize_over_inclination in enumerate([False, True]):
    get_loglike = get_loglike_function(ydeg, marginalize_over_inclination)
    for j in range(len(npts)):
        tstart = time.time()
        t = np.linspace(0, 1, npts[j])
        for k in range(ncalls):
            get_loglike(t)
        tloglike[i, j] = (time.time() - tstart) / ncalls

# Plot
fig, ax = plt.subplots(1)
ax.plot(npts, tloglike[0], label="conditional")
ax.plot(npts, tloglike[1], label="marginal")

# Asymptotic
n = np.array([2e2, 1.0e4])
ax.plot(n, 1.2e-10 * n ** 2.6, "k--", alpha=0.5)
ax.annotate(
    r"$\propto K^{2.6}$",
    fontsize=16,
    xy=(2e3, 1e-2),
    xycoords="data",
    xytext=(0, 0),
    textcoords="offset points",
    ha="center",
    va="center",
    rotation=35,
    alpha=0.5,
)

# Appearance
ax.margins(0, 0)
ax.legend(fontsize=12)
ax.set_xlabel("number of points")
ax.set_ylabel("likelihood eval time [s]")
ax.set_yscale("log")
ax.set_xscale("log")

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
