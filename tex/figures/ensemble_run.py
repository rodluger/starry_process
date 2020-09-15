from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
from corner import corner
import theano.tensor as tt
import os


# Don't run this script on Azure
if int(os.environ.get("ON_AZURE", 0)):
    import sys

    sys.exit(0)


# Load the data
data = np.load("ensemble_data.npz")
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]
period = data["period"]

# DEBUG: Just two light curves
flux = flux[:2]

np.random.seed(0)
with pm.Model() as model:

    # Priors
    sa = pm.Uniform("sa", 0, 1)
    sb = pm.Uniform("sb", 0, 1)
    la = pm.Uniform("la", 0, 1)
    lb = pm.Uniform("lb", 0, 1)
    ca = pm.Uniform("ca", 0, 1)
    cb = pm.Uniform("cb", 0, 1)

    # Set up the GP
    sp = StarryProcess(sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, period=period)

    # Compute the likelihood for each light curve
    inc = [None for k in range(len(flux))]
    for k in range(len(flux)):

        # Allow a different inclination for each light curve
        # The inclination is distributed according to p(i) = sin(i)
        inc[k] = pm.Uniform("inc_{:02d}".format(k), 0, 0.5 * np.pi)
        pm.Potential("sin_{:02d}".format(k), tt.log(tt.abs_(tt.sin(inc[k]))))

        # Go under the hood to change the inclination
        # NOTE: Internally, starry process uses radians!
        sp.design._inc = inc[k]

        # The log likelihood, marginalized over all maps
        pm.Potential(
            "marginal_{:02d}".format(k),
            sp.log_likelihood(t, flux[k], ferr ** 2, baseline_mean=1.0),
        )

        # The Jacobian, to enforce a uniform prior over the mean
        # and standard deviations of the spot sizes & latitudes
        pm.Potential("jacobian_{:02d}".format(k), sp.log_jac())

    # Optimize
    map_soln = xo.optimize(start=model.test_point)

    # Sample
    print("Sampling...")
    trace = pm.sample(
        tune=500,
        draws=1000,
        chains=4,
        start=map_soln,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )

    breakpoint()
    pass
