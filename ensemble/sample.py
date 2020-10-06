from .generate import generate
import starry
from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
import theano
import theano.tensor as tt
import os
from corner import corner
import pandas as pd
from scipy.signal import medfilt
import json


PATH = os.path.abspath(os.path.dirname(__file__))


def lat2y(lat):
    """
    Return the fractional y position (in [0, 1])
    corresponding to a given latitude on a Mollweide grid.
    
    """
    theta = lat * np.pi / 180
    niter = 100
    for n in range(niter):
        theta -= (2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * theta)
        )
    return np.sin(theta)


def sample(runid, clobber=False, debug=False):

    # Get input kwargs
    INPUT_FILE = os.path.join(PATH, "{:02d}".format(runid), "input.json")
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("sample", {})
    method = inputs.get("method", "advi")
    nadvi = inputs.get("nadvi", 25000)
    nadvi_samples = inputs.get("nadvi_samples", 100000)
    nuts_tune = inputs.get("nuts_tune", 500)
    nuts_draws = inputs.get("nuts_draws", 2000)
    nuts_chains = inputs.get("nuts_chains", 4)
    baseline_var = inputs.get("baseline_var", 1e-2)
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("generate", {})
    ydeg_true = inputs.get("ydeg", 30)

    # File names
    DATA_FILE = os.path.join(PATH, "{:02d}".format(runid), "data.npz")
    TRUTH_FILE = os.path.join(PATH, "{:02d}".format(runid), "truth.npz")
    SAMPLES_FILE = os.path.join(
        PATH, "{:02d}".format(runid), "{}_samples.pkl".format(method)
    )
    HIST_FILE = os.path.join(
        PATH, "{:02d}".format(runid), "{}_hist.npz".format(method)
    )
    DATA_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_data.pdf".format(method)
    )
    CORNER_DIAGNOSTIC_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_corner_diagnostic.pdf".format(method)
    )
    HIST_DIAGNOSTIC_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_hist_diagnostic.pdf".format(method)
    )
    INC_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_inc.pdf".format(method)
    )
    CORNER_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_corner.pdf".format(method)
    )
    POSTERIOR_SAMPLES_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_posterior_samples.pdf".format(method)
    )
    POSTERIOR_SAMPLES_NO_NULLSPACE_PLOT = os.path.join(
        PATH,
        "{:02d}".format(runid),
        "{}_posterior_samples_no_nullspace.pdf".format(method),
    )

    # Run the sampler
    if clobber or not os.path.exists(SAMPLES_FILE):

        # Generate (and save) the data
        data, truth = generate(runid)
        np.savez(DATA_FILE, **data)
        np.savez(TRUTH_FILE, **truth)
        t = data["t"]
        flux = data["flux"]
        ferr = data["ferr"]
        nlc = len(flux)

        # Set up the model
        np.random.seed(0)
        if not debug:
            with pm.Model() as model:

                # Priors
                s = pm.Uniform("s", 10, 30, testval=truth["s"])
                la = pm.Uniform("la", 0, 1, testval=truth["la"])
                lb = pm.Uniform("lb", 0, 1, testval=truth["lb"])
                c = pm.Uniform("c", 0, 1, testval=truth["c"])
                N = pm.Uniform("N", 1, 50, testval=truth["N"])
                incs = pm.Uniform(
                    "incs", 0, 90, shape=(nlc,), testval=truth["incs"]
                )
                periods = truth["periods"]

                # Set up the GP
                sp = StarryProcess(s=s, la=la, lb=lb, c=c, N=N)

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
                pm.Potential(
                    "sini", tt.sum(tt.log(tt.sin(incs * np.pi / 180)))
                )
                pm.Potential("jacobian", sp.log_jac())

                if method == "advi":

                    # Fit
                    print("Fitting...")
                    advi_fit = pm.fit(
                        n=nadvi,
                        method=pm.FullRankADVI(),
                        random_seed=0,
                        start=model.test_point,
                        callbacks=[
                            pm.callbacks.CheckParametersConvergence(
                                diff="relative"
                            )
                        ],
                    )

                    # Sample
                    print("Sampling...")
                    trace = advi_fit.sample(nadvi_samples)
                    samples = pm.trace_to_dataframe(trace)

                    # Save the loss history
                    hist = advi_fit.hist
                    np.savez(HIST_FILE, hist=hist)

                elif method == "nuts":

                    print("Sampling...")
                    trace = pm.sample(
                        tune=nuts_tune,
                        draws=nuts_draws,
                        start=model.test_point,
                        chains=nuts_chains,
                        step=xo.get_dense_nuts_step(target_accept=0.9),
                    )
                    samples = pm.trace_to_dataframe(trace)

                else:

                    raise ValueError("invalid method")

                # Display the summary
                print(pm.summary(trace))

                # Pickle the trace
                samples.to_pickle(SAMPLES_FILE)

    # Plotting
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("plot", {})
    vmin = inputs.get("vmin", 0.73)
    vmax = inputs.get("vmax", 1.02)
    ndraws = inputs.get("ndraws", 5)
    res = inputs.get("res", 300)

    # Load everything we'll need for plotting
    if not debug:
        samples = pd.read_pickle(SAMPLES_FILE)
        if method == "advi":
            hist = np.load(HIST_FILE)["hist"]
    else:
        samples = pd.DataFrame()
        for param in ["s", "la", "lb", "c", "N"]:
            samples[param] = np.random.random(1000)
        for param in ["incs__{:d}".format(i) for i in range(nlc)]:
            samples[param] = 90 * np.random.random(1000)
        if method == "advi":
            hist = np.random.randn(1000)

    data = np.load(DATA_FILE)
    truth = np.load(TRUTH_FILE)
    t = data["t"]
    flux0 = data["flux0"]
    flux = data["flux"]
    ferr = data["ferr"]
    nlc = len(flux)
    nsamples = len(samples)

    # Plot the synthetic dataset
    if (nlc % 10) == 0:
        nrows = nlc // 10
    else:
        nrows = 1
    wr = np.ones(nlc // nrows)
    wr[-1] = 1.17
    gridspec = {"width_ratios": wr}
    fig, ax = plt.subplots(
        2 * nrows, nlc // nrows, figsize=(12, 2 * nrows), gridspec_kw=gridspec,
    )
    fig.subplots_adjust(hspace=0.4)
    axtop = ax.transpose().flatten()[::2]
    axbot = ax.transpose().flatten()[1::2]
    yrng = 1.1 * np.max(np.abs(1e3 * (flux0)))
    ymin = -yrng
    ymax = yrng
    xe = 2 * np.linspace(-1, 1, 1000)
    ye = np.sqrt(1 - (0.5 * xe) ** 2)
    eps = 0.01
    xe = (1 - eps) * xe
    ye = (1 - 0.5 * eps) * ye
    for k in range(nlc):
        im = axtop[k].imshow(
            truth["images"][k],
            origin="lower",
            extent=(-2, 2, -1, 1),
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        axtop[k].plot(xe, ye, "k-", lw=1, clip_on=False)
        axtop[k].plot(xe, -ye, "k-", lw=1, clip_on=False)
        axtop[k].plot(0, lat2y(90 - truth["incs"][k]), "kx", ms=3)
        axtop[k].axis("off")
        axbot[k].plot(t, 1e3 * (flux[k]), "k.", alpha=0.3, ms=1)
        axbot[k].plot(t, 1e3 * (flux0[k]), "C0-", lw=1)
        axbot[k].set_ylim(ymin, ymax)
        if k < nrows:
            div = make_axes_locatable(axtop[nlc - k - 1])
            cax = div.append_axes("right", size="7%", pad="10%")
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label("intensity", fontsize=8)
            cbar.set_ticks([0.75, 1])
            cbar.ax.tick_params(labelsize=6)
            axbot[k].spines["top"].set_visible(False)
            axbot[k].spines["right"].set_visible(False)
            axbot[k].set_xlabel("rotations", fontsize=8)
            axbot[k].set_ylabel("flux [ppt]", fontsize=8)
            axbot[k].set_xticks([0, 1, 2, 3, 4])
            for tick in (
                axbot[k].xaxis.get_major_ticks()
                + axbot[k].yaxis.get_major_ticks()
            ):
                tick.label.set_fontsize(6)
            axbot[k].tick_params(direction="in")
        else:
            axbot[k].axis("off")
    fig.savefig(DATA_PLOT, bbox_inches="tight", dpi=300)
    plt.close()

    # Diagnostic plots
    varnames = ["s", "la", "lb", "c", "N"]
    truths = [truth[v] for v in varnames]
    varnames += ["incs__{:d}".format(k) for k in range(nlc)]
    truths += list(truth["incs"])
    fig = corner(samples[varnames], truths=truths)
    fig.savefig(CORNER_DIAGNOSTIC_PLOT, bbox_inches="tight")
    if method == "advi":
        fig, ax = plt.subplots(1)
        lh = np.log10(hist - np.min(hist) + 1)
        ax.plot(range(len(lh)), lh)
        w = 299
        ax.plot(
            range(len(lh))[w // 2 : -w // 2], medfilt(lh, w)[w // 2 : -w // 2]
        )
        ax.set_ylabel("relative log loss")
        ax.set_xlabel("iteration number")
        fig.savefig(HIST_DIAGNOSTIC_PLOT, bbox_inches="tight")
        plt.close()

    # Inclination histograms
    nrows = int(np.ceil(nlc / 5))
    fig, ax = plt.subplots(nrows, 5, figsize=(16, 2.5 * nrows), sharex=True)
    bins = np.linspace(0, 90, 50)
    for k, axis in enumerate(ax.flatten()):
        if k >= nlc:
            axis.axis("off")
            continue
        axis.hist(
            samples["incs__{:d}".format(k)],
            bins=bins,
            histtype="step",
            color="k",
        )
        axis.axvline(truth["incs"][k])
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
    fig.savefig(INC_PLOT, bbox_inches="tight")
    plt.close()

    # Draw posterior samples
    theano.config.compute_test_value = "off"
    _draw = lambda flux, s, la, lb, c, N, inc, period: StarryProcess(
        s=s, la=la, lb=lb, c=c, N=N, inc=inc, period=period
    ).sample_ylm_conditional(t, flux, ferr ** 2, baseline_var=baseline_var)
    _s = tt.dscalar()
    _la = tt.dscalar()
    _lb = tt.dscalar()
    _c = tt.dscalar()
    _N = tt.dscalar()
    _inc = tt.dscalar()
    _period = tt.dscalar()
    _flux = tt.dvector()
    draw = theano.function(
        [_flux, _s, _la, _lb, _c, _N, _inc, _period],
        _draw(_flux, _s, _la, _lb, _c, _N, _inc, _period),
    )
    ydeg_inf = 15
    image_pred = np.empty((nlc, ndraws, res, res))
    image_pred_no_null = np.empty((nlc, ndraws, res, res))
    flux_pred = np.empty((nlc, ndraws, len(t)))
    incs_pred = np.empty((nlc, ndraws))
    map = starry.Map(ydeg_inf, lazy=False)
    for n in tqdm(range(nlc)):
        for i in range(ndraws):
            idx = np.random.choice(nsamples)
            y = draw(
                flux[n],
                samples["s"][idx],
                samples["la"][idx],
                samples["lb"][idx],
                samples["c"][idx],
                samples["N"][idx],
                samples["incs__{:d}".format(n)][idx],
                truth["periods"][n],
            )
            map[:, :] = y
            map.inc = samples["incs__{:d}".format(n)][idx]

            # Flux sample
            flux_pred[n, i] = map.flux(theta=360 / truth["periods"][n] * t)

            # Image sample
            image_pred[n, i] = 1.0 + map.render(projection="moll", res=res)

            # Image sample (null space zeroed out)
            for l in range(3, ydeg_inf + 1, 2):
                map[l, :] = 0
            image_pred_no_null[n, i] = 1.0 + map.render(
                projection="moll", res=res
            )

            # The background intensity is not an observable, so let's
            # renormalize so it's unity
            for img in [image_pred, image_pred_no_null]:
                hist, edges = np.histogram(
                    img[n, i].flatten(),
                    bins=100,
                    range=(np.nanmin(img[n, i]), np.nanmax(img[n, i]),),
                )
                j = np.argmax(hist)
                bkg = 0.5 * (edges[j] + edges[j + 1])
                img[n, i] += 1 - bkg

            incs_pred[n, i] = samples["incs__{:d}".format(n)][idx]

    # Get the true images w/ and w/out null space
    map_true = starry.Map(ydeg_true, lazy=False)
    image = np.empty((nlc, res, res))
    image_no_null = np.empty((nlc, res, res))
    for n in tqdm(range(nlc)):
        map_true[:, :] = truth["y"][n]
        image[n] = 1.0 + map_true.render(projection="moll", res=res)
        for l in range(3, ydeg_true + 1, 2):
            map_true[l, :] = 0
        image_no_null[n] = 1.0 + map_true.render(projection="moll", res=res)

    # Show the map & flux samples
    for img, img_pred, FILE_NAME in zip(
        [image, image_no_null],
        [image_pred, image_pred_no_null],
        [POSTERIOR_SAMPLES_PLOT, POSTERIOR_SAMPLES_NO_NULLSPACE_PLOT],
    ):
        fig, ax = plt.subplots(
            1 + ndraws, nlc, figsize=(12 * (1 + (nlc - 1) // 10), (1 + ndraws))
        )
        for axis in ax.flatten():
            axis.axis("off")
        fig.subplots_adjust(bottom=0.25)
        axflux = [None for n in range(nlc)]
        for n in range(nlc):
            pos = ax[-1, n].get_position()
            left = pos.x0
            width = pos.width
            bottom = pos.y0 - (ax[-2, n].get_position().y0 - pos.y0)
            height = pos.height
            axflux[n] = fig.add_axes([left, bottom, width, height])

        xe = 2 * np.linspace(-1, 1, 1000)
        ye = np.sqrt(1 - (0.5 * xe) ** 2)
        eps = 0.02
        xe = 0.5 * eps + (1 - eps) * xe
        ye = 0.5 * eps + (1 - 0.5 * eps) * ye
        yrng = 1.1 * np.max(np.abs(1e3 * (flux0)))
        ymin = -yrng
        ymax = yrng
        for n in range(nlc):
            # True map
            ax[0, n].imshow(
                img[n],
                origin="lower",
                extent=(-2, 2, -1, 1),
                cmap="plasma",
                vmin=vmin,
                vmax=vmax,
            )
            ax[0, n].plot(xe, ye, "k-", lw=1, clip_on=False)
            ax[0, n].plot(xe, -ye, "k-", lw=1, clip_on=False)
            ax[0, n].plot(0, lat2y(90 - truth["incs"][n]), "kx", ms=3)

            # Map samples
            for i in range(ndraws):
                ax[1 + i, n].imshow(
                    img_pred[n, i],
                    origin="lower",
                    extent=(-2, 2, -1, 1),
                    cmap="plasma",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax[1 + i, n].plot(xe, ye, "k-", lw=1, clip_on=False)
                ax[1 + i, n].plot(xe, -ye, "k-", lw=1, clip_on=False)
                ax[1 + i, n].plot(
                    0, lat2y(90 - truth["incs"][n]), "kx", ms=3, alpha=0.5,
                )
                ax[1 + i, n].plot(0, lat2y(90 - incs_pred[n, i]), "kx", ms=3)

            # True flux
            axflux[n].plot(
                t, 1e3 * (flux[n] - np.median(flux[n])), "k.", alpha=0.3, ms=1
            )
            axflux[n].set_ylim(ymin, ymax)

            # Flux samples
            for i in range(ndraws):
                axflux[n].plot(
                    t,
                    1e3 * (flux_pred[n, i] - np.median(flux_pred[n, i])),
                    "C0-",
                    lw=1,
                    alpha=0.5,
                )

            if n == 0:
                axflux[n].spines["top"].set_visible(False)
                axflux[n].spines["right"].set_visible(False)
                axflux[n].set_xlabel("rotations", fontsize=8)
                axflux[n].set_ylabel("flux [ppt]", fontsize=8)
                axflux[n].set_xticks([0, 1, 2, 3, 4])
                for tick in (
                    axflux[n].xaxis.get_major_ticks()
                    + axflux[n].yaxis.get_major_ticks()
                ):
                    tick.label.set_fontsize(6)
                axflux[n].tick_params(direction="in")
            else:
                axflux[n].axis("off")

        fig.savefig(FILE_NAME, bbox_inches="tight", dpi=300)
        plt.close()
