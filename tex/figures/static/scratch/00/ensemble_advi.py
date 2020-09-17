from starry_process import StarryProcess
from starry_process.math import cast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
from corner import corner
import theano
import theano.tensor as tt
import os


# Options
optimize = False
nadvi = 10000
noptim = 1000
nsamples = 10000
seed = 0
compute = False
nlc = -1


# Load the data
FILE = os.path.abspath(__file__)
data = np.load(__file__.replace(".py", "_data.npz"))
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]
periods_true = data["periods"]

# Keep only `nlc` light curves
flux = flux[:nlc]
ferr = ferr[:nlc]
periods_true = periods_true[:nlc]

# Number of light curves
nlc = len(flux)

# Number of parameters
nvars = 6 + nlc

# Let's go
if compute:

    with pm.Model() as model:

        # Priors
        sa = pm.Uniform("sa", 0, 1)
        sb = pm.Uniform("sb", 0, 1)
        la = pm.Uniform("la", 0, 1)
        lb = pm.Uniform("lb", 0, 1)
        ca = pm.Uniform("ca", 0, 1)
        cb = pm.Uniform("cb", 0, 1)

        # Set up the GP
        sp = StarryProcess(sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb)

        # Compute the likelihood for each light curve
        incs = [None for k in range(nlc)]
        for k in range(nlc):

            # Allow a different inclination for each light curve
            # The inclination is distributed according to p(i) = sin(i)
            incs[k] = pm.Uniform("inc_{:02d}".format(k), 0, 0.5 * np.pi)
            pm.Potential(
                "sin_{:02d}".format(k), tt.log(tt.abs_(tt.sin(incs[k])))
            )

            # Let's assume we know the true period for simplicity
            omega = cast(2 * np.pi / periods_true[k])

            # Go under the hood to change the inclination and the period
            sp.design._omega = omega  # radians / day
            sp.design._inc = incs[k]  # radians

            # The log likelihood, marginalized over all maps
            pm.Potential(
                "marginal_{:02d}".format(k),
                sp.log_likelihood(t, flux[k], ferr[k] ** 2, baseline_mean=1.0),
            )

            # The Jacobian, to enforce a uniform prior over the mean
            # and standard deviations of the spot sizes & latitudes
            pm.Potential("jacobian_{:02d}".format(k), sp.log_jac())

        # Optimize
        if optimize:
            print("Optimizing...")
            map_soln = xo.optimize(
                start=model.test_point, options=dict(maxiter=noptim)
            )
        else:
            map_soln = model.test_point

        # Fit
        print("Fitting...")
        advi_fit = pm.fit(
            n=nadvi, method=pm.ADVI(), random_seed=seed, start=map_soln
        )

        # Sample
        print("Sampling...")
        advi_trace = advi_fit.sample(nsamples)

        # Display the summary
        print(pm.summary(advi_trace))

        # Transform to params we care about
        samples = np.empty((nvars, nsamples))
        samples[4] = advi_trace["ca"]
        samples[5] = advi_trace["cb"]
        samples[6:] = [
            advi_trace["inc_{:02d}".format(k)] * 180 / np.pi
            for k in range(nlc)
        ]
        for k in tqdm(range(nsamples)):
            samples[0, k], samples[1, k] = sp.size.transform.inverse_transform(
                advi_trace["sa"][k], advi_trace["sb"][k]
            )
            (
                samples[2, k],
                samples[3, k],
            ) = sp.latitude.transform.inverse_transform(
                advi_trace["la"][k], advi_trace["lb"][k]
            )

        # Save the samples
        np.savez_compressed(
            FILE.replace(".py", "_samples.npz"), samples=samples
        )

else:

    samples = np.load(FILE.replace(".py", "_samples.npz"))["samples"]

# Transformed variable names
varnames = [
    r"$\mu_s$",
    r"$\sigma_s$",
    r"$\mu_l$",
    r"$\sigma_l$",
    r"$\mu_c$",
    r"$\sigma_c$",
]
varnames += ["$i_{{{:02d}}}$".format(k) for k in range(nlc)]

# True values
truths = [
    data["rmu"][:nlc],
    data["rsig"][:nlc],
    data["lmu"][:nlc],
    data["lsig"][:nlc],
    data["cmu"][:nlc],
    data["csig"][:nlc],
]
truths += list(data["incs"][:nlc])

# Bounds
bounds = [(0, 90), (0, 50), (0, 90), (0, 50), (0, 1), (0, 1)]
bounds += [(0, 90) for k in range(nlc)]

# X axis ticks
xticks = [
    (0, 15, 30, 45, 60, 75, 90),
    (0, 10, 20, 30, 40, 50),
    (0, 15, 30, 45, 60, 75, 90),
    (0, 10, 20, 30, 40, 50),
    (0, 0.25, 0.5, 0.75, 1.0),
    (0, 0.25, 0.5, 0.75, 1.0),
]
xticks += [(0, 15, 30, 45, 60, 75, 90) for k in range(nlc)]

# Plot the 1d posteriors
fig, ax = plt.subplots(1, 6, figsize=(12, 1.5))
for k in range(6):
    ax[k].hist(
        samples[k],
        bins=np.linspace(xticks[k][0], xticks[k][-1], 50),
        histtype="step",
        color="k",
    )
    ax[k].axvline(truths[k], color="C0", lw=1)
    ax[k].set_yticks([])
    ax[k].set_xlabel(varnames[k])
    ax[k].set_xlim(bounds[k])
    ax[k].set_xticks(xticks[k])
    for tick in ax[k].xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation(45)
fig.savefig(FILE.replace(".py", ".pdf"), bbox_inches="tight")

# Plot the 1d inclination posteriors
fig, ax = plt.subplots(1, nlc, figsize=(2 * nlc, 1.5))
ax = np.atleast_1d(ax)
for k in range(nlc):
    ax[k].hist(
        samples[6 + k],
        bins=np.linspace(xticks[6 + k][0], xticks[6 + k][-1], 50),
        histtype="step",
        color="k",
    )
    ax[k].axvline(truths[6 + k], color="C0", lw=1)
    ax[k].set_yticks([])
    ax[k].set_xlabel(varnames[6 + k])
    ax[k].set_xlim(bounds[6 + k])
    ax[k].set_xticks(xticks[6 + k])
    for tick in ax[k].xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation(45)
fig.savefig(FILE.replace(".py", "_inc.pdf"), bbox_inches="tight")
