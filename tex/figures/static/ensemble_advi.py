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
nsamples = 10000
seed = 0

# Directory for saving the trace
TRACE_DIR = os.path.abspath(__file__).replace(".py", "_trace")

# Load the data
data = np.load("ensemble_data.npz")
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]
periods_true = data["periods"]

# Number of light curves
nlc = len(flux)

# Number of parameters
nvars = 6 + nlc

# Let's go
try:
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
                start=model.test_point, options=dict(maxiter=10)
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

        # Save the trace
        pm.save_trace(advi_trace, directory=TRACE_DIR, overwrite=True)

        # Quickplot of the trace
        ax = pm.traceplot(advi_trace)
        fig = ax.flatten()[0].figure
        fig.savefig(
            __file__.replace(".py", "_traceplot.pdf"), bbox_inches="tight"
        )

        # Transform to params we care about
        varnames = [
            "r$\mu_s$",
            "r$\sig_s$",
            "r$\mu_l$",
            "r$\sig_l$",
            "r$\mu_c$",
            "r$\sig_c$",
        ]
        varnames += ["$i_{{{:02d}}}$".format(k) for k in range(nlc)]
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

        # True values
        truths = [
            data["rmu"],
            data["rsig"],
            data["lmu"],
            data["lsig"],
            data["cmu"],
            data["csig"],
        ]
        truths += list(data["incs"])

        # Plot the 1d posteriors
        fig, ax = plt.subplots(1, 6, figsize=(12, 1.5))
        for k in range(6):
            ax[k].hist(samples[k], bins=30, histtype="step", color="k")
            ax[k].axvline(truths[k], color="C0", lw=1, alpha=0.5)
            ax[k].set_yticks([])
            ax[k].set_xlabel(varnames[k])
            for tick in ax[k].xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
        fig.savefig(__file__.replace(".py", ".pdf"), bbox_inches="tight")

        # Plot the 1d inclination posteriors
        fig, ax = plt.subplots(1, nlc, figsize=(2 * nlc, 1.5))
        ax = np.atleast_1d(ax)
        for k in range(nlc):
            ax[k].hist(samples[6 + k], bins=30, histtype="step", color="k")
            ax[k].axvline(truths[6 + k], color="C0", lw=1, alpha=0.5)
            ax[k].set_yticks([])
            ax[k].set_xlabel(varnames[6 + k])
            for tick in ax[k].xaxis.get_major_ticks():
                tick.label.set_fontsize(10)
        fig.savefig(__file__.replace(".py", "_inc.pdf"), bbox_inches="tight")

        # DEBUG
        breakpoint()
        pass

except:

    breakpoint()
    pass
