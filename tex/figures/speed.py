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


ydegs = [5, 10, 15]
npts = np.array(np.logspace(0, 4, 20), dtype=int)
ncalls = 10
tloglike = np.zeros((len(ydegs), len(npts)))

for i in tqdm(range(len(ydegs)), disable=bool(int(os.getenv("NOTQDM", "0")))):
    get_loglike = get_loglike_function(ydegs[i], False)
    for j in range(len(npts)):
        tstart = time.time()
        t = np.linspace(0, 1, npts[j])
        for k in range(ncalls):
            get_loglike(t)
        tloglike[i, j] = (time.time() - tstart) / ncalls


fig, ax = plt.subplots(1)
for i in range(len(ydegs)):
    ax.plot(
        npts, tloglike[i], label=r"$l_\mathrm{{max}} = {}$".format(ydegs[i])
    )

ax.legend(fontsize=10)
ax.set_xlabel("number of points")
ax.set_ylabel("likelihood evaluation time [s]")
ax.set_yscale("log")
ax.set_xscale("log")

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"),
    bbox_inches="tight",
    dpi=300,
)
