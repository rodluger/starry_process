from .generate import generate
import starry
from starry_process import StarryProcess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.lines as lines
from tqdm import tqdm
import pymc3 as pm
import exoplanet as xo
import theano
import theano.tensor as tt
import os
from corner import corner
import pandas as pd
from scipy.signal import medfilt
from scipy.stats import norm as Normal
import json


PATH = os.path.abspath(os.path.dirname(__file__))


def lat2y(lat):
    """
    Return the fractional y position (in [0, 1])
    corresponding to a given latitude on a Mollweide grid.
    
    """
    lat = lat * np.pi / 180
    theta = lat
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
    fit_for_s_var = inputs.get("fit_for_s_var", False)
    fit_for_l_var = inputs.get("fit_for_l_var", True)
    good_guess = inputs.get("good_guess", True)
    smin = inputs.get("spot_size_min", 10.0)
    smax = inputs.get("spot_size_max", 30.0)
    with open(INPUT_FILE, "r") as f:
        inputs = json.load(f).get("generate", {})
    ydeg_true = inputs.get("ydeg", 30)
    varnames = []
    labels = []
    if fit_for_s_var:
        varnames += ["sa", "sb"]
        labels += [r"$\alpha_{r}$", r"$\beta_{r}$"]
    else:
        varnames += ["s"]
        labels += ["radius [deg]"]
    if fit_for_l_var:
        varnames += ["la", "lb"]
        labels += [r"$\alpha_{\phi}$", r"$\beta_{\phi}$"]
    else:
        varnames += ["l"]
        labels += ["latitude [deg]"]
    varnames += ["c", "N"]
    labels += ["contrast", r"$N$"]

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
    LATITUDE_PDF_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_latitude.pdf".format(method)
    )
    POSTERIOR_SAMPLES_PLOT = os.path.join(
        PATH, "{:02d}".format(runid), "{}_posterior_samples.pdf".format(method)
    )
    POSTERIOR_SAMPLES_NO_NULLSPACE_PLOT = os.path.join(
        PATH,
        "{:02d}".format(runid),
        "{}_posterior_samples_no_nullspace.pdf".format(method),
    )
    POSTERIOR_SAMPLES_NO_GP_PLOT = os.path.join(
        PATH,
        "{:02d}".format(runid),
        "{}_posterior_samples_no_gp.pdf".format(method),
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

        breakpoint()

        # Set up the model
        np.random.seed(0)
        if not debug:
            with pm.Model() as model:

                # Priors
                if fit_for_s_var:
                    sa = pm.Uniform(
                        "sa", 0, 1, testval=truth["sa"] if good_guess else 0.5
                    )
                    sb = pm.Uniform(
                        "sb", 0, 1, testval=truth["sb"] if good_guess else 0.5
                    )
                    size = [sa, sb]
                else:
                    s = pm.Uniform(
                        "s",
                        smin,
                        smax,
                        testval=max(min(truth["s"], smax), smin)
                        if good_guess
                        else 0.5 * (smin + smax),
                    )
                    size = [s]
                if fit_for_l_var:
                    la = pm.Uniform(
                        "la", 0, 1, testval=truth["la"] if good_guess else 0.5
                    )
                    lb = pm.Uniform(
                        "lb", 0, 1, testval=truth["lb"] if good_guess else 0.5
                    )
                    latitude = [la, lb]
                else:
                    l = pm.Uniform(
                        "l", 0, 90, testval=truth["l"] if good_guess else 45
                    )
                    latitude = [l]
                c = pm.Uniform(
                    "c", 0, 1, testval=truth["c"] if good_guess else 0.5
                )
                N = pm.Uniform(
                    "N", 1, 50, testval=truth["N"] if good_guess else 10
                )
                contrast = [c, N]
                incs = pm.Uniform(
                    "incs",
                    0,
                    90,
                    shape=(nlc,),
                    testval=truth["incs"]
                    if good_guess
                    else 60.0 * np.ones(nlc),
                )
                periods = truth["periods"]

                # Set up the GP
                sp = StarryProcess(
                    size=size, latitude=latitude, contrast=contrast
                )

                # Likelihood for each light curve
                log_like = []
                for k in range(nlc):
                    log_like.append(
                        sp.log_likelihood(
                            t,
                            flux[k],
                            ferr ** 2,
                            baseline_var=baseline_var,
                            period=periods[k],
                            inc=incs[k],
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
    La = inputs.get("uninformative_prior_a", 1.0)
    Ls = inputs.get("uninformative_prior_s", 2.0)
    Lm = inputs.get("uninformative_prior_m", 3.0)
    L0 = inputs.get("uninformative_prior_0", 1e-2)
    Lfac = inputs.get("uninformative_prior_brightening", 1.5)

    # Load everything we'll need for plotting
    if not debug:
        samples = pd.read_pickle(SAMPLES_FILE)
        if method == "advi":
            hist = np.load(HIST_FILE)["hist"]
    else:
        samples = pd.DataFrame()
        for param in varnames:
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
    varnames_all = list(varnames)
    truths = [truth[v] for v in varnames_all]
    varnames_all += ["incs__{:d}".format(k) for k in range(nlc)]
    truths += list(truth["incs"])
    fig = corner(samples[varnames_all], truths=truths)
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

    # Corner plot
    truths = [truth[v] for v in varnames]
    fig = corner(samples[varnames], truths=truths, labels=labels)
    fig.savefig(CORNER_PLOT, bbox_inches="tight")

    # Inclination histograms
    nrows = int(np.ceil(nlc / 5))
    fig, ax = plt.subplots(nrows, 5, figsize=(16, 2.5 * nrows), sharex=True)
    bins = np.linspace(0, 90, 100)
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
        xticks = [0, 30, 60, 90]
        axis.set_xticks(xticks)
        axis.set_xticklabels(["{:d}$^\circ$".format(xt) for xt in xticks])
        if k >= 5:
            axis.set_xlabel("inclination")
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

    # Latitude PDF samples
    if fit_for_l_var:
        theano.config.compute_test_value = "off"
        _draw_pdf = lambda x, la, lb: StarryProcess(
            latitude=[la, lb]
        ).latitude.pdf(x)
        _x = tt.dvector()
        _la = tt.dscalar()
        _lb = tt.dscalar()
        draw_pdf = theano.function([_x, _la, _lb], _draw_pdf(_x, _la, _lb),)
        nx = 1000
        npdf = 1000
        x = np.linspace(-89.9, 89.9, nx)
        pdf_true = 0.5 * (
            Normal.pdf(x, truth["lmu"], truth["lsig"])
            + Normal.pdf(x, -truth["lmu"], truth["lsig"])
        )
        pdf = np.empty((npdf, nx))
        for k in range(npdf):
            idx = np.random.choice(npdf)
            pdf[k] = draw_pdf(x, samples["la"][idx], samples["lb"][idx],)
        fig, ax = plt.subplots(1)
        for k in range(npdf):
            ax.plot(x, pdf[k], "C0-", lw=1, alpha=0.1)
        ax.plot(x, pdf_true, "C1-", label="truth")
        ax.plot(x, np.nan * pdf_true, "C0-", label="samples")
        ax.legend(loc="upper right")
        ax.set_xlim(-90, 90)
        xticks = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:d}$^\circ$".format(xt) for xt in xticks])
        ax.set_xlabel("latitude")
        ax.set_ylabel("probability")
        fig.savefig(LATITUDE_PDF_PLOT, bbox_inches="tight")
        plt.close()

    # Draw posterior samples
    theano.config.compute_test_value = "off"
    _s = tt.dscalar()
    _sa = tt.dscalar()
    _sb = tt.dscalar()
    _l = tt.dscalar()
    _la = tt.dscalar()
    _lb = tt.dscalar()
    _c = tt.dscalar()
    _N = tt.dscalar()
    _inc = tt.dscalar()
    _period = tt.dscalar()
    _flux = tt.dvector()
    if fit_for_s_var and fit_for_l_var:
        _draw = lambda flux, sa, sb, la, lb, c, N, inc, period: StarryProcess(
            size=[sa, sb], latitude=[la, lb], contrast=[c, N]
        ).sample_ylm_conditional(
            t,
            flux,
            ferr ** 2,
            baseline_var=baseline_var,
            period=period,
            inc=inc,
        )
        draw = theano.function(
            [_flux, _sa, _sb, _la, _lb, _c, _N, _inc, _period],
            _draw(_flux, _sa, _sb, _la, _lb, _c, _N, _inc, _period),
        )
    elif fit_for_s_var:
        _draw = lambda flux, sa, sb, l, c, N, inc, period: StarryProcess(
            size=[sa, sb], latitude=l, contrast=[c, N]
        ).sample_ylm_conditional(
            t,
            flux,
            ferr ** 2,
            baseline_var=baseline_var,
            period=period,
            inc=inc,
        )
        draw = theano.function(
            [_flux, _sa, _sb, _l, _c, _N, _inc, _period],
            _draw(_flux, _sa, _sb, _l, _c, _N, _inc, _period),
        )
    elif fit_for_l_var:
        _draw = lambda flux, s, la, lb, c, N, inc, period: StarryProcess(
            size=s, latitude=[la, lb], contrast=[c, N]
        ).sample_ylm_conditional(
            t,
            flux,
            ferr ** 2,
            baseline_var=baseline_var,
            period=period,
            inc=inc,
        )
        draw = theano.function(
            [_flux, _s, _la, _lb, _c, _N, _inc, _period],
            _draw(_flux, _s, _la, _lb, _c, _N, _inc, _period),
        )
    else:
        _draw = lambda flux, s, l, c, N, inc, period: StarryProcess(
            size=s, latitude=l, contrast=[c, N]
        ).sample_ylm_conditional(
            t,
            flux,
            ferr ** 2,
            baseline_var=baseline_var,
            period=period,
            inc=inc,
        )
        draw = theano.function(
            [_flux, _s, _l, _c, _N, _inc, _period],
            _draw(_flux, _s, _l, _c, _N, _inc, _period),
        )
    ydeg_inf = 15
    image_pred = np.empty((nlc, ndraws, res, res))
    image_pred_no_null = np.empty((nlc, ndraws, res, res))
    image_pred_no_gp = np.empty((nlc, ndraws, res, res))
    flux_pred = np.empty((nlc, ndraws, len(t)))
    flux_pred_no_gp = np.empty((nlc, ndraws, len(t)))
    incs_pred = np.empty((nlc, ndraws))
    map = starry.Map(ydeg_inf, lazy=False)
    for n in tqdm(range(nlc)):
        for i in range(ndraws):
            idx = np.random.choice(nsamples)
            incs_pred[n, i] = samples["incs__{:d}".format(n)][idx]
            y = draw(
                flux[n],
                *[samples[v][idx] for v in varnames],
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

            # Image & flux samples (uninformative prior, for comparison)
            L = np.concatenate(
                [
                    La * np.exp(-(((l - Lm) / Ls) ** 2)) * np.ones(2 * l + 1)
                    for l in range(ydeg_inf + 1)
                ]
            )
            L[0] = L0
            map.reset(inc=truth["incs"][n])
            map.set_data(1 + flux[n], C=ferr ** 2)
            map.set_prior(L=L)
            map.solve(theta=360.0 / truth["periods"][n] * t)
            map.draw()
            flux_pred_no_gp[n, i] = map.flux(
                theta=360 / truth["periods"][n] * t
            )
            image_pred_no_gp[n, i] = 1.0 + map.render(
                projection="moll", res=res
            )

            # Hack: The point with these plots is to show how poorly an
            # uninformative prior does. In general, the brightness/contrast
            # is going to be VERY wrong, so if we were to plot the maps on
            # the same brightness scale, we wouldn't see ANYTHING. Here we
            # fudge the overall contrast to make it look better and highlight
            # where the dark and bright regions are on the map.
            image_pred_no_gp[n, i] -= np.nanmin(image_pred_no_gp[n, i])
            image_pred_no_gp[n, i] /= np.nanmax(image_pred_no_gp[n, i])
            image_pred_no_gp[n, i] *= Lfac * (vmax - vmin)
            image_pred_no_gp[n, i] += vmin

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
    for img, img_pred, show_inc, flx_pred, FILE_NAME in zip(
        [image, image_no_null, image],
        [image_pred, image_pred_no_null, image_pred_no_gp],
        [True, True, False],
        [flux_pred, flux_pred, flux_pred_no_gp],
        [
            POSTERIOR_SAMPLES_PLOT,
            POSTERIOR_SAMPLES_NO_NULLSPACE_PLOT,
            POSTERIOR_SAMPLES_NO_GP_PLOT,
        ],
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
            bottom -= 0.025
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
                if show_inc:
                    ax[1 + i, n].plot(
                        0, lat2y(90 - incs_pred[n, i]), "kx", ms=3
                    )

            # True flux
            axflux[n].plot(
                t, 1e3 * (flux[n] - np.median(flux[n])), "k.", alpha=0.3, ms=1
            )
            axflux[n].set_ylim(ymin, ymax)

            # Flux samples
            for i in range(ndraws):
                axflux[n].plot(
                    t,
                    1e3 * (flx_pred[n, i] - np.median(flx_pred[n, i])),
                    "C0-",
                    lw=1,
                    alpha=0.5,
                )

            if n == 0:
                axflux[n].spines["top"].set_visible(False)
                axflux[n].spines["right"].set_visible(False)
                axflux[n].set_xlabel("rotations", fontsize=6)
                axflux[n].set_xticks([0, 1, 2, 3, 4])
                for tick in (
                    axflux[n].xaxis.get_major_ticks()
                    + axflux[n].yaxis.get_major_ticks()
                ):
                    tick.label.set_fontsize(6)
                axflux[n].tick_params(direction="in")
            else:
                axflux[n].axis("off")

        fig.add_artist(
            lines.Line2D(
                [0.10, 0.91], [0.781, 0.781], color="k", lw=1, alpha=0.5
            )
        )
        fig.add_artist(
            lines.Line2D(
                [0.10, 0.10], [0.10, 0.88], color="k", lw=1, alpha=0.5
            )
        )
        fig.add_artist(
            lines.Line2D(
                [0.10, 0.91], [0.24, 0.24], color="k", lw=1, alpha=0.5
            )
        )

        plt.figtext(
            0.075, 0.39, "posterior samples", rotation="vertical", fontsize=12
        )
        plt.figtext(0.075, 0.81, "truth", rotation="vertical", fontsize=12)
        plt.figtext(0.075, 0.15, "flux", rotation="vertical", fontsize=12)

        fig.savefig(FILE_NAME, bbox_inches="tight", dpi=300)
        plt.close()
