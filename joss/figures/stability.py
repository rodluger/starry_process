import numpy as np
import matplotlib.pyplot as plt
from starry_process import StarryProcess, gauss2beta
import theano
import theano.tensor as tt
from tqdm import tqdm
import os

# Compile the function to get the covariance matrix
r = tt.dscalar()
a = tt.dscalar()
b = tt.dscalar()
c = tt.dscalar()
n = tt.dscalar()
get_cov = theano.function(
    [r, a, b, c, n],
    StarryProcess(
        ydeg=20, epsy=1e-12, epsy15=0, r=r, a=a, b=b, c=c, n=n
    ).cov_ylm,
)

# Plot the condition number for 100 prior samples
C = lambda cov, l: np.linalg.cond(cov[: (l + 1) ** 2, : (l + 1) ** 2])
ls = np.arange(1, 21)
nsamples = 100

fig, ax = plt.subplots(1)
np.random.seed(0)
for j in tqdm(range(nsamples), disable=bool(int(os.getenv("NOTQDM", "0")))):
    r = np.random.uniform(10, 45)
    c = np.random.random()
    n = np.random.uniform(1, 50)
    mu = np.random.uniform(0, 85)
    sigma = np.random.uniform(5, 40)
    cov = get_cov(r, *gauss2beta(mu, sigma), c, n)
    logC = np.log10([C(cov, l) for l in ls])
    ax.plot(ls, logC, color="C0", lw=1, alpha=0.25)

ax.margins(0, None)
ax.axvline(15, color="k", lw=1, ls="--")
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel(r"$l_\mathrm{max}$", fontsize=18)
ax.set_ylabel(r"log$_{10}$ condition number")

# We're done!
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
