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
import pickle


# Options
nadvi = 10000
noptim = 0
nsamples = 10000
seed = 0
compute = True
nlc = -1
baseline_var = 1e-5

# Load the data
FILE = os.path.abspath(__file__)
data = np.load(__file__.replace(".py", "_data.npz"))
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]
periods_true = data["periods"]
incs_true = data["incs"]

# Keep only `nlc` light curves
flux = flux[:nlc]
ferr = ferr[:nlc]
periods_true = periods_true[:nlc]
incs_true = incs_true[:nlc]
nlc = len(flux)
nvars = 5 + nlc

# Let's go
if compute:

    with pm.Model() as model:

        # Priors
        sa = pm.Uniform("sa", 0, 1)
        sb = pm.Uniform("sb", 0, 1)
        la = pm.Uniform("la", 0, 1)
        lb = pm.Uniform("lb", 0, 1)
        ca = pm.Uniform("ca", 0, 5)
        cb = 0.0
        incs = pm.Uniform("incs", 0, 0.5 * np.pi, shape=(nlc,))
        pm.Potential("sini", tt.sum(tt.log(tt.sin(incs))))
        periods = periods_true

        # Set up the GP
        sp = StarryProcess(sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb)
        pm.Potential("jac", nlc * sp.log_jac())

        # Sum the log likelihoods for each light curve
        for k in range(nlc):
            sp.design._omega = cast(2 * np.pi / periods[k])
            sp.design._inc = incs[k]
            pm.Potential(
                "marginal_{:02d}".format(k),
                sp.log_likelihood(
                    t,
                    flux[k],
                    ferr[k] ** 2,
                    baseline_mean=1.0,
                    baseline_var=baseline_var,
                ),
            )

        # Optimize
        if noptim > 0:
            print("Optimizing...")
            map_soln = xo.optimize(
                start=model.test_point, options=dict(maxiter=noptim)
            )
        else:
            map_soln = model.test_point

        # Fit
        print("Fitting...")
        advi_fit = pm.fit(
            n=nadvi,
            method=pm.FullRankADVI(),
            random_seed=seed,
            start=map_soln,
            callbacks=[
                pm.callbacks.CheckParametersConvergence(diff="relative")
            ],
        )

        # Display the loss history
        fig, ax = plt.subplots(1)
        ax.plot(advi_fit.hist)
        ax.set_xlabel("iteration number")
        ax.set_ylabel("negative ELBO")
        fig.savefig(FILE.replace(".py", "_hist.pdf"), bbox_inches="tight")

        # Sample
        print("Sampling...")
        advi_trace = advi_fit.sample(nsamples)

        # Display the summary
        print(pm.summary(advi_trace))

        # Transform to params we care about
        print("Transforming...")
        samples = np.empty((nsamples, nvars))
        samples[:, 4] = advi_trace["ca"]
        samples[:, 5:] = advi_trace["incs"] * 180 / np.pi
        for n in tqdm(range(nsamples)):
            samples[n, 0:2] = sp.size.transform.inverse_transform(
                advi_trace["sa"][n], advi_trace["sb"][n]
            )
            samples[n, 2:4] = sp.latitude.transform.inverse_transform(
                advi_trace["la"][n], advi_trace["lb"][n]
            )

        # Save the samples
        np.savez_compressed(
            FILE.replace(".py", "_samples.npz"), samples=samples
        )

else:

    samples = np.load(FILE.replace(".py", "_samples.npz"))["samples"]


# Transformed variable names
varnames = [r"$\mu_s$", r"$\sigma_s$", r"$\mu_l$", r"$\sigma_l$", r"$\mu_c$"]
varnames += ["$i_{{{:02d}}}$".format(k) for k in range(nlc)]

# True values
truths = [data["rmu"], data["rsig"], data["lmu"], data["lsig"], data["cmu"]]
truths += list(data["incs"][:nlc])

# Bounds
bounds = [(0, 90), (0, 50), (0, 90), (0, 50), (0, 5)]
bounds += [(0, 90) for k in range(nlc)]

# X axis ticks
xticks = [
    (0, 15, 30, 45, 60, 75, 90),
    (0, 10, 20, 30, 40, 50),
    (0, 15, 30, 45, 60, 75, 90),
    (0, 10, 20, 30, 40, 50),
    (0, 1, 2, 3, 4, 5),
]
xticks += [(0, 15, 30, 45, 60, 75, 90) for k in range(nlc)]

# Plot the 1d posteriors
fig, ax = plt.subplots(1, nvars, figsize=(12, 1.5))
for k in range(nvars):
    ax[k].hist(
        samples[:, k],
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
        samples[:, 5 + k],
        bins=np.linspace(xticks[5 + k][0], xticks[5 + k][-1], 50),
        histtype="step",
        color="k",
    )
    ax[k].axvline(truths[5 + k], color="C0", lw=1)
    ax[k].set_yticks([])
    ax[k].set_xlabel(varnames[5 + k])
    ax[k].set_xlim(bounds[5 + k])
    ax[k].set_xticks(xticks[5 + k])
    for tick in ax[k].xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation(45)
fig.savefig(FILE.replace(".py", "_inc.pdf"), bbox_inches="tight")

# Plot the corner plot
fig = corner(samples, labels=varnames, truths=truths)
fig.savefig(FILE.replace(".py", "_corner.pdf"), bbox_inches="tight")
