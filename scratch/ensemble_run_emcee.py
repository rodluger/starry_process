from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import theano
import emcee
from corner import corner
import theano.tensor as tt
import os
import sys

# Don't run this script on Azure
if int(os.environ.get("ON_AZURE", 0)):
    sys.exit(0)

# Directory for saving the trace
TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chains")
if not os.path.exists(TRACE_DIR):
    os.mkdir(TRACE_DIR)

# Load the data
data = np.load("ensemble_data.npz")
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]
period = data["period"]

# DEBUG: Only 5 light curves
flux = flux[:5]

# Compile the theano functions
_sa = tt.dscalar()
_sb = tt.dscalar()
_la = tt.dscalar()
_lb = tt.dscalar()
_ca = tt.dscalar()
_cb = tt.dscalar()
_inc = tt.dscalar()
_flux = tt.dvector()
_ferr = tt.dscalar()
log_jac = theano.function(
    [_sa, _sb, _la, _lb],
    StarryProcess(sa=_sa, sb=_sb, la=_la, lb=_lb).log_jac(),
)
log_likelihood = theano.function(
    [_sa, _sb, _la, _lb, _ca, _cb, _inc, _flux, _ferr],
    StarryProcess(
        sa=_sa, sb=_sb, la=_la, lb=_lb, ca=_ca, cb=_cb, inc=_inc, period=period
    ).log_likelihood(t, _flux, _ferr ** 2, baseline_mean=1.0),
)


def log_prob(x):
    # All params
    sa, sb, la, lb, ca, cb, *incs = x

    # Uniform priors
    for p in [sa, sb, la, lb, ca, cb]:
        if p < 0 or p > 1:
            return -np.inf
    for inc in incs:
        if inc < 0 or inc > 90:
            return -np.inf

    # Inclination priors
    ll = np.sum(np.log(np.abs(np.sin(inc))))

    # Jacobian transform for lat & size
    ll += log_jac(sa, sb, la, lb)

    # GP likelihood
    for k in range(len(flux)):
        ll += log_likelihood(sa, sb, la, lb, ca, cb, incs[k], flux[k], ferr)

    return ll


ndim = 6 + len(flux)
nwalkers = 2 * ndim

# Sample from the prior
p0 = np.random.uniform(size=(nwalkers, ndim))
p0[:, 6:] = np.arccos(p0[:, 6:]) * 180 / np.pi

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(p0, 10000, progress=True)

