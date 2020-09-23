from generate import generate
import starry
from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
import theano
import theano.tensor as tt
import os
from corner import corner
import pandas as pd
from scipy.signal import medfilt

FILE = os.path.abspath(__file__)
SAMPLES_FILE = FILE.replace(".py", "_samples.pkl")
HIST_FILE = FILE.replace(".py", "_hist.npz")

# Options
nlc = 10
nadvi = 25000
nsamples = 100000
seed = 0
baseline_var = 1e-2
clobber = False
np.random.seed(0)

# Get the data
data, truth, fig = generate(nlc=nlc, plot=True)
fig.savefig(FILE.replace(".py", "_data.pdf"), bbox_inches="tight")
t = data["t"]
flux = data["flux"]
ferr = data["ferr"]

# Run
if clobber or not os.path.exists(SAMPLES_FILE):

    # Set up the model
    with pm.Model() as model:

        # Priors
        sa = pm.Uniform("sa", 0, 1, testval=truth["sa"])
        sb = pm.Uniform("sb", 0, 1, testval=truth["sb"])
        la = pm.Uniform("la", 0, 1, testval=truth["la"])
        lb = pm.Uniform("lb", 0, 1, testval=truth["lb"])
        ca = pm.Uniform("ca", 0, 5, testval=truth["ca"])
        cb = truth["cb"]
        incs = pm.Uniform(
            "incs", 0, 0.5 * np.pi, shape=(nlc,), testval=truth["incs"]
        )
        periods = truth["periods"]

        # Set up the GP
        sp = StarryProcess(
            sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, angle_unit="rad"
        )

        # Likelihood for each light curve
        log_like = []
        for k in range(nlc):
            sp.design._set_params(period=periods[k], inc=incs[k])
            log_like.append(
                sp.log_likelihood(
                    t, flux[k], ferr ** 2, baseline_var=baseline_var
                )
            )
        pm.Potential("marginal", tt.sum(log_like))

        # Priors
        pm.Potential("sini", tt.sum(tt.log(tt.sin(incs))))
        pm.Potential("jacobian", sp.log_jac())

        # Fit
        print("Fitting...")
        advi_fit = pm.fit(
            n=nadvi,
            method=pm.FullRankADVI(),
            random_seed=seed,
            start=model.test_point,
            callbacks=[
                pm.callbacks.CheckParametersConvergence(diff="relative")
            ],
        )

        # Sample
        print("Sampling...")
        trace = advi_fit.sample(nsamples)
        samples = pm.trace_to_dataframe(trace)

        # Display the summary
        print(pm.summary(trace))

        # Transform to physical parameters
        samples["smu"] = np.empty_like(samples["sa"])
        samples["ssig"] = np.empty_like(samples["sa"])
        samples["lmu"] = np.empty_like(samples["sa"])
        samples["lsig"] = np.empty_like(samples["sa"])
        for n in tqdm(range(nsamples)):
            (
                samples["smu"][n],
                samples["ssig"][n],
            ) = sp.size.transform.inverse_transform(
                samples["sa"][n], samples["sb"][n]
            )
            (
                samples["lmu"][n],
                samples["lsig"][n],
            ) = sp.latitude.transform.inverse_transform(
                samples["la"][n], samples["lb"][n]
            )

        # Pickle the trace
        samples.to_pickle(SAMPLES_FILE)
        hist = advi_fit.hist
        np.savez(HIST_FILE, hist=hist)

else:

    # Load the trace
    samples = pd.read_pickle(SAMPLES_FILE)
    hist = np.load(HIST_FILE)["hist"]


# Diagnostic plots
varnames = ["sa", "sb", "la", "lb", "ca"]
truths = [truth[v] for v in varnames]
varnames += ["incs__{:d}".format(k) for k in range(nlc)]
truths += list(truth["incs"])
fig = corner(samples[varnames], truths=truths)
fig.savefig(FILE.replace(".py", "_corner_diagnostic.pdf"), bbox_inches="tight")
fig, ax = plt.subplots(1)
lh = np.log10(hist - np.min(hist) + 1)
ax.plot(range(len(lh)), lh)
w = 299
ax.plot(range(len(lh))[w // 2 : -w // 2], medfilt(lh, w)[w // 2 : -w // 2])
ax.set_ylabel("relative log loss")
ax.set_xlabel("iteration number")
fig.savefig(FILE.replace(".py", "_hist_diagnostic.pdf"), bbox_inches="tight")

# Inclination histograms
fig, ax = plt.subplots(2, 5, figsize=(16, 5), sharex=True)
bins = np.linspace(0, 90, 50)
for k, axis in enumerate(ax.flatten()):
    axis.hist(
        samples["incs__{:d}".format(k)] * 180 / np.pi,
        bins=bins,
        histtype="step",
        color="k",
    )
    axis.axvline(truth["incs"][k] * 180 / np.pi)
    axis.set_yticks([])
    axis.set_xticks([0, 30, 60, 90])
    if k >= 5:
        axis.set_xlabel("inclination [deg]")
    axis.annotate(
        "{:d}".format(k + 1),
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=12,
    )
fig.savefig(FILE.replace(".py", "_inc.pdf"), bbox_inches="tight")

# Corner plot
varnames = ["smu", "ssig", "lmu", "lsig", "ca"]
labels = [
    r"$\mu_{\Delta\theta}$",
    r"$\sigma_{\Delta\theta}$",
    r"$\mu_{\phi}$",
    r"$\sigma_{\phi}$",
    r"$\xi$",
]
fig = corner(
    samples[varnames],
    truths=[truth[v] for v in varnames],
    labels=labels,
    range=[0.99, 0.99, 1, 1, 1],
)
fig.savefig(FILE.replace(".py", "_corner.pdf"), bbox_inches="tight")

# Draw posterior samples
ndraws = 5
_draw = lambda flux, sa, sb, la, lb, ca: tt.reshape(
    StarryProcess(
        sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=0
    ).sample_ylm_conditional(t, flux, ferr ** 2, baseline_var=baseline_var),
    (-1,),
)
_sa = tt.dscalar()
_sb = tt.dscalar()
_la = tt.dscalar()
_lb = tt.dscalar()
_ca = tt.dscalar()
_flux = tt.dvector()
theano.config.compute_test_value = "ignore"
draw = theano.function(
    [_flux, _sa, _sb, _la, _lb, _ca], _draw(_flux, _sa, _sb, _la, _lb, _ca)
)
ydeg = 15
inds = np.random.choice(nsamples, replace=False, size=ndraws)
y = np.empty((nlc, ndraws, (ydeg + 1) ** 2))
for n in tqdm(range(nlc)):
    for i in range(ndraws):
        y[n, i] = draw(
            flux[n],
            samples["sa"][inds[i]],
            samples["sb"][inds[i]],
            samples["la"][inds[i]],
            samples["lb"][inds[i]],
            samples["ca"][inds[i]],
        )

# Show the map samples
fig, ax = plt.subplots(1 + ndraws, nlc, figsize=(12, (1 + ndraws)))
for axis in ax.flatten():
    axis.axis("off")
map = starry.Map(ydeg, lazy=False)
flux_samples = np.empty((nlc, ndraws, len(t)))
xe = 2 * np.linspace(-1, 1, 1000)
ye = np.sqrt(1 - (0.5 * xe) ** 2)
eps = 0.02
xe = 0.5 * eps + (1 - eps) * xe
ye = 0.5 * eps + (1 - 0.5 * eps) * ye
for n in range(nlc):
    ax[0, n].imshow(
        truth["images"][n],
        origin="lower",
        extent=(-2, 2, -1, 1),
        cmap="plasma",
        vmin=0.5,
        vmax=1.1,
    )
    ax[0, n].plot(xe, ye, "k-", lw=1, clip_on=False)
    ax[0, n].plot(xe, -ye, "k-", lw=1, clip_on=False)
    for i in range(ndraws):
        map[:, :] = y[n, i]
        map.inc = samples["incs__{:d}".format(n)][inds[i]] * 180 / np.pi
        image = 1.0 + map.render(projection="moll", res=150)
        ax[1 + i, n].imshow(
            image,
            origin="lower",
            extent=(-2, 2, -1, 1),
            cmap="plasma",
            vmin=0.5,
            vmax=1.1,
        )
        ax[1 + i, n].plot(xe, ye, "k-", lw=1, clip_on=False)
        ax[1 + i, n].plot(xe, -ye, "k-", lw=1, clip_on=False)
        flux_samples[n, i] = map.flux(theta=360 / truth["periods"][n] * t)
fig.savefig(FILE.replace(".py", "_map_samples.pdf"), bbox_inches="tight")

# Show the flux samples
fig, ax = plt.subplots(1, nlc, figsize=(12, 1), sharex=True, sharey=True)
yrng = 1.1 * np.max(np.abs(1e3 * (flux)))
ymin = -yrng
ymax = yrng
for n, axis in enumerate(ax):
    axis.plot(t, 1e3 * flux[n], "k.", alpha=0.3, ms=1)
    for i in range(ndraws):
        axis.plot(
            t,
            (
                1e3
                * (
                    (1 + flux_samples[n, i])
                    / (np.median(1 + flux_samples[n, i]))
                    - 1
                )
            ),
            "C0-",
            lw=1,
            alpha=0.75,
        )
    axis.set_ylim(ymin, ymax)
    if n == 0:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("rotations", fontsize=8)
        axis.set_ylabel("flux [ppt]", fontsize=8)
        axis.set_xticks([0, 1, 2, 3, 4])
        for tick in (
            axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks()
        ):
            tick.label.set_fontsize(6)
        axis.tick_params(direction="in")
    else:
        axis.axis("off")
fig.savefig(FILE.replace(".py", "_flux_samples.pdf"), bbox_inches="tight")
