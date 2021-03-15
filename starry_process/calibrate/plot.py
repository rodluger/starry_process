from .defaults import update_with_defaults
from .log_prob import get_log_prob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from starry_process import StarryProcess
from starry_process.latitude import beta2gauss, gauss2beta
import starry
from starry_process.compat import theano, tt
import numpy as np
from scipy.stats import norm as Normal
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
from corner import corner as _corner
from tqdm.auto import tqdm
import glob
import os
import json
import pickle


def corner(*args, **kwargs):
    """
    Override `corner.corner` by making some appearance tweaks.

    """
    # Get the usual corner plot
    figure = _corner(*args, **kwargs)

    # Get the axes
    ndim = int(np.sqrt(len(figure.axes)))
    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Smaller tick labels
    for ax in axes[1:, 0]:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(
            ax.get_ylabel(), fontsize=kwargs.get("corner_label_size", 16)
        )
    for ax in axes[-1, :]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(
            ax.get_xlabel(), fontsize=kwargs.get("corner_label_size", 16)
        )

    # Pad the axes to always include the truths
    truths = kwargs.get("truths", None)
    if truths is not None:
        for row in range(1, ndim):
            for col in range(row):
                lo, hi = np.array(axes[row, col].get_xlim())
                if truths[col] < lo:
                    lo = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)
                elif truths[col] > hi:
                    hi = truths[col] - 0.1 * (hi - truths[col])
                    axes[row, col].set_xlim(lo, hi)
                    axes[col, col].set_xlim(lo, hi)

                lo, hi = np.array(axes[row, col].get_ylim())
                if truths[row] < lo:
                    lo = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)
                elif truths[row] > hi:
                    hi = truths[row] - 0.1 * (hi - truths[row])
                    axes[row, col].set_ylim(lo, hi)
                    axes[row, row].set_xlim(lo, hi)

    return figure


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


def plot_data(data, ncols=10, clip=False, **kwargs):
    """
    Plot a synthetic dataset.

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    plot_kwargs = kwargs["plot"]
    vmin = plot_kwargs["vmin"]
    vmax = plot_kwargs["vmax"]

    # Get data
    t = data["t"]
    incs = data["incs"]
    flux = data["flux"]
    flux0 = data["flux0"]
    y = data["y"]

    # Plot the synthetic dataset
    nlc = len(flux)
    if clip:
        nlc = divmod(nlc, ncols)[0] * ncols
    if nlc > ncols:
        nrows = int(np.ceil(nlc / ncols))
    else:
        nrows = 1
    wr = np.ones(min(nlc, ncols))
    wr[-1] = 1.17
    gridspec = {"width_ratios": wr}
    fig, ax = plt.subplots(
        2 * nrows,
        min(nlc, ncols),
        figsize=(min(nlc, ncols) + 2, 2 * nrows),
        gridspec_kw=gridspec,
    )
    fig.subplots_adjust(hspace=0.4)
    axtop = ax.transpose().flatten()[::2]
    axbot = ax.transpose().flatten()[1::2]
    yrng = 1.1 * np.max(
        np.abs(1e3 * (flux0 - np.median(flux0, axis=1).reshape(-1, 1)))
    )
    ymin = -yrng
    ymax = yrng
    xe = 2 * np.linspace(-1, 1, 1000)
    ye = np.sqrt(1 - (0.5 * xe) ** 2)
    eps = 0.01
    xe = (1 - eps) * xe
    ye = (1 - 0.5 * eps) * ye
    map = starry.Map(kwargs["generate"]["ydeg"], lazy=False)
    for k in tqdm(range(nlc), disable=bool(int(os.getenv("NOTQDM", "0")))):
        map[:, :] = y[k]
        image = 1.0 + map.render(projection="moll", res=300)
        im = axtop[k].imshow(
            image,
            origin="lower",
            extent=(-2, 2, -1, 1),
            cmap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        axtop[k].plot(xe, ye, "k-", lw=1, clip_on=False)
        axtop[k].plot(xe, -ye, "k-", lw=1, clip_on=False)
        axtop[k].plot(0, lat2y(90 - incs[k]), "kx", ms=3)
        axtop[k].axis("off")
        axtop[k].set_ylim(-1.01, 2.0)
        axbot[k].plot(
            t, 1e3 * (flux[k] - np.median(flux[k])), "k.", alpha=0.3, ms=1
        )
        axbot[k].plot(t, 1e3 * (flux0[k] - np.median(flux0[k])), "C0-", lw=1)
        axbot[k].set_ylim(ymin, ymax)
        if k < nrows:
            axins = axtop[nlc - k - 1].inset_axes([0, 0, 1, 0.67])
            axins.axis("off")
            div = make_axes_locatable(axins)
            cax = div.append_axes("right", size="7%", pad="50%")
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

    for k in range(nlc, len(axtop)):
        axtop[k].axis("off")
        axbot[k].axis("off")

    for axis in ax.flatten():
        axis.set_rasterization_zorder(99)

    return fig


def plot_latitude_pdf(results, **kwargs):
    """
    Plot posterior draws from the latitude hyperdistribution.

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    plot_kwargs = kwargs["plot"]
    gen_kwargs = kwargs["generate"]
    mu_true = gen_kwargs["latitude"]["mu"]
    sigma_true = gen_kwargs["latitude"]["sigma"]
    nlat_pts = plot_kwargs["nlat_pts"]
    nlat_samples = plot_kwargs["nlat_samples"]

    # Resample to equal weight
    samples = np.array(results.samples)
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]
    samples = dyfunc.resample_equal(samples, weights)

    # Function to compute the pdf for a draw
    _draw_pdf = lambda x, a, b: StarryProcess(a=a, b=b).latitude.pdf(x)
    _x = tt.dvector()
    _a = tt.dscalar()
    _b = tt.dscalar()

    # The true pdf
    draw_pdf = theano.function([_x, _a, _b], _draw_pdf(_x, _a, _b))
    x = np.linspace(-89.9, 89.9, nlat_pts)
    if np.isfinite(sigma_true):
        pdf_true = 0.5 * (
            Normal.pdf(x, mu_true, sigma_true)
            + Normal.pdf(x, -mu_true, sigma_true)
        )
    else:
        # Isotropic (special case)
        pdf_true = 0.5 * np.cos(x * np.pi / 180) * np.pi / 180

    # Draw sample pdfs
    pdf = np.empty((nlat_samples, nlat_pts))
    for k in range(nlat_samples):
        idx = np.random.randint(len(samples))
        pdf[k] = draw_pdf(x, samples[idx, 1], samples[idx, 2])

    # Plot
    fig, ax = plt.subplots(1)
    for k in range(nlat_samples):
        ax.plot(x, pdf[k], "C0-", lw=1, alpha=0.05, zorder=-1)
    ax.plot(x, pdf_true, "C1-", label="truth")
    ax.plot(x, np.nan * x, "C0-", label="samples")
    ax.legend(loc="upper right")
    ax.set_xlim(-90, 90)
    xticks = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:d}$^\circ$".format(xt) for xt in xticks])
    ax.set_xlabel("latitude", fontsize=16)
    ax.set_ylabel("probability", fontsize=16)
    # Constrain y lims?
    mx1 = np.max(pdf_true)
    mx2 = np.sort(pdf.flatten())[int(0.9 * len(pdf.flatten()))]
    mx = max(2.0 * mx1, 1.2 * mx2)
    ax.set_ylim(-0.1 * mx, mx)
    ax.set_rasterization_zorder(1)
    return fig


def plot_trace(results, **kwargs):
    """
    Plot the nested sampling trace.

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    gen_kwargs = kwargs["generate"]
    labels = ["r", "a", "b", "c", "n", "bm", "blv"]

    # Get truths
    try:
        a, b = StarryProcess().latitude._transform.transform(
            gen_kwargs["latitude"]["mu"], gen_kwargs["latitude"]["sigma"]
        )
    except:
        a = np.nan
        b = np.nan
    truths = [
        gen_kwargs["radius"]["mu"],
        a,
        b,
        gen_kwargs["contrast"]["mu"],
        gen_kwargs["nspots"]["mu"],
        np.nan,
        np.nan,
    ]
    ndim = results.samples.shape[-1]
    fig, _ = dyplot.traceplot(
        results, truths=truths[:ndim], labels=labels[:ndim]
    )
    return fig


def plot_corner(results, transform_beta=False, **kwargs):
    """
    Plot the posterior corner plot.

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    gen_kwargs = kwargs["generate"]
    sample_kwargs = kwargs["sample"]
    plot_kwargs = kwargs["plot"]
    span = [
        (sample_kwargs["rmin"], sample_kwargs["rmax"]),
        (sample_kwargs["amin"], sample_kwargs["amax"]),
        (sample_kwargs["bmin"], sample_kwargs["bmax"]),
        (sample_kwargs["cmin"], sample_kwargs["cmax"]),
        (sample_kwargs["nmin"], sample_kwargs["nmax"]),
        0.995,
        0.995,
    ]
    labels = [
        r"$r$",
        r"$a$",
        r"$b$",
        r"$c$",
        r"$n$",
        r"$\mu_b$",
        r"$\ln\sigma^2_b$",
    ]

    # Get truths
    sp = StarryProcess()
    try:
        a, b = gauss2beta(
            gen_kwargs["latitude"]["mu"], gen_kwargs["latitude"]["sigma"]
        )
    except:
        a = np.nan
        b = np.nan
    truths = [
        gen_kwargs["radius"]["mu"],
        a,
        b,
        gen_kwargs["contrast"]["mu"],
        gen_kwargs["nspots"]["mu"],
        np.nan,
        np.nan,
    ]

    samples = np.array(results.samples)
    ndim = samples.shape[-1]

    if transform_beta:

        # Transform from `a, b` to `mode, std`
        a = samples[:, 1]
        b = samples[:, 2]
        mu, sigma = beta2gauss(a, b)
        samples[:, 1] = mu
        samples[:, 2] = sigma
        labels[1] = r"$\mu_\phi$"
        labels[2] = r"$\sigma_\phi$"
        if np.isfinite(gen_kwargs["latitude"]["sigma"]):
            truths[1] = gen_kwargs["latitude"]["mu"]
            truths[2] = gen_kwargs["latitude"]["sigma"]
        else:
            truths[1] = np.nan
            truths[2] = np.nan
        span[1] = (0, 90)
        span[2] = (0, 45)

    # Get sample weights
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]

    fig = corner(
        samples[:, :ndim],
        plot_datapoints=False,
        plot_density=False,
        truths=truths[:ndim],
        labels=labels[:ndim],
        range=span[:ndim],
        fill_contours=True,
        weights=weights,
        smooth=2.0,
        smooth1d=2.0,
        bins=100,
        hist_kwargs=dict(lw=1),
        truth_color="#ff7f0e",
        **plot_kwargs
    )

    return fig


def plot_inclination_pdf(data, inc_results, **kwargs):
    # Get the arrays
    inc = inc_results["inc"]
    lp = inc_results["lp"]

    # Plot
    nlc = lp.shape[0]
    if nlc > 10:
        nrows = int(np.ceil(nlc / 10))
    else:
        nrows = 1
    fig, ax = plt.subplots(
        nrows,
        min(nlc, 10),
        figsize=(min(nlc, 10) + 2, nrows),
        sharex=True,
        sharey=True,
    )
    ax = ax.flatten()
    for n in range(lp.shape[0]):
        for j in range(lp.shape[1]):
            ax[n].plot(
                inc, np.exp(lp[n, j] - lp[n, j].max()), "C0-", lw=1, alpha=0.25
            )
        ax[n].axvline(data["incs"][n], color="C1")
        ax[n].margins(0.1, 0.1)
        if n == 40:
            ax[n].spines["top"].set_visible(False)
            ax[n].spines["right"].set_visible(False)
            ax[n].set_xlabel("inclination", fontsize=10)
            ax[n].set_ylabel("probability", fontsize=10)
            xticks = [0, 30, 60, 90]
            ax[n].set_xticks(xticks)
            ax[n].set_xticklabels([r"{}$^\circ$".format(xt) for xt in xticks])
            ax[n].set_yticks([])
            for tick in ax[n].xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
        else:
            ax[n].axis("off")
    return fig


def plot_batch(path):
    """
    Plot the results of a batch run.

    """
    # Get the posterior means and covariances (w/o baseline mean and var)
    files = glob.glob(os.path.join(path, "*", "mean_and_cov.npz"))
    mean = np.empty((len(files), 5))
    cov = np.empty((len(files), 5, 5))
    for k, file in enumerate(files):
        data = np.load(file)
        mean[k] = data["mean"][:5]
        cov[k] = data["cov"][:5, :5]

    # Get the true values
    kwargs = update_with_defaults(
        **json.load(
            open(files[0].replace("mean_and_cov.npz", "kwargs.json"), "r")
        )
    )
    truths = [
        kwargs["generate"]["radius"]["mu"],
        kwargs["generate"]["latitude"]["mu"],
        kwargs["generate"]["latitude"]["sigma"],
        kwargs["generate"]["contrast"]["mu"],
        kwargs["generate"]["nspots"]["mu"],
    ]
    labels = [r"$r$", r"$\mu_\phi$", r"$\sigma_\phi$", r"$c$", r"$n$"]

    # Misc plotting kwargs
    batch_bins = kwargs["plot"]["batch_bins"]
    batch_alpha = kwargs["plot"]["batch_alpha"]
    batch_nsig = kwargs["plot"]["batch_nsig"]

    # Plot the distribution of posterior means & variances
    fig, ax = plt.subplots(
        2,
        len(truths) + 1,
        figsize=(16, 6),
        gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 0.01]),
    )
    fig.subplots_adjust(hspace=0.4)
    for n in range(len(truths)):

        # Distribution of means
        ax[0, n].hist(
            mean[:, n], histtype="step", bins=batch_bins, lw=2, density=True
        )
        ax[0, n].axvline(np.mean(mean[:, n]), color="C0", ls="--")
        ax[0, n].axvline(truths[n], color="C1")

        # Distribution of errors (should be ~ std normal)
        deltas = (mean[:, n] - truths[n]) / np.sqrt(cov[:, n, n])
        ax[1, n].hist(
            deltas,
            density=True,
            histtype="step",
            range=(-4, 4),
            bins=batch_bins,
            lw=2,
        )
        ax[1, n].hist(
            np.random.randn(10000),
            density=True,
            range=(-4, 4),
            bins=batch_bins,
            histtype="step",
            lw=2,
        )

        ax[0, n].set_title(labels[n], fontsize=16)
        ax[0, n].set_xlabel("posterior mean")
        ax[1, n].set_xlabel("posterior error")
        ax[0, n].set_yticks([])
        ax[1, n].set_yticks([])

    # Tweak appearance
    ax[0, -1].axis("off")
    ax[0, -1].plot(0, 0, "C0", ls="--", label="mean")
    ax[0, -1].plot(0, 0, "C1", ls="-", label="truth")
    ax[0, -1].legend(loc="center left")
    ax[1, -1].axis("off")
    ax[1, -1].plot(0, 0, "C0", ls="-", lw=2, label="measured")
    ax[1, -1].plot(0, 0, "C1", ls="-", lw=2, label=r"$\mathcal{N}(0, 1)$")
    ax[1, -1].legend(loc="center left")
    fig.savefig(
        os.path.join(path, "calibration_bias.pdf"), bbox_inches="tight"
    )
    plt.close()

    # Now let's plot all of the posteriors on a corner plot
    files = glob.glob(os.path.join(path, "*", "results.pkl"))
    samples = [None for k in range(len(files))]
    nsamp = 1e9
    ranges = [None for k in range(len(files))]
    for k in tqdm(
        range(len(files)), disable=bool(int(os.getenv("NOTQDM", "0")))
    ):

        # Get the samples (w/o baseline mean and var)
        with open(files[k], "rb") as f:
            results = pickle.load(f)
        samples[k] = np.array(results.samples)[:, :5]
        samples[k][:, 1], samples[k][:, 2] = beta2gauss(
            samples[k][:, 1], samples[k][:, 2]
        )
        try:
            weights = np.exp(results["logwt"] - results["logz"][-1])
        except:
            weights = results["weights"]
        samples[k] = dyfunc.resample_equal(samples[k], weights)
        np.random.shuffle(samples[k])
        if len(samples[k]) < nsamp:
            nsamp = len(samples[k])

        # Get the 4-sigma ranges
        mu = np.mean(samples[k], axis=0)
        std = np.std(samples[k], axis=0)
        ranges[k] = np.array([mu - batch_nsig * std, mu + batch_nsig * std]).T

    # We need all runs to have the same number of samples
    # so our normalizations are correct in the histograms
    print("Keeping {} samples from each run.".format(nsamp))

    # Set plot limits to the maximum of the ranges
    ranges = np.array(ranges)
    ranges = np.array(
        [np.min(ranges[:, :, 0], axis=0), np.max(ranges[:, :, 1], axis=0)]
    ).T

    span = [
        (kwargs["sample"]["rmin"], kwargs["sample"]["rmax"]),
        (0, 90),
        (0, 45),
        (kwargs["sample"]["cmin"], kwargs["sample"]["cmax"]),
        (kwargs["sample"]["nmin"], kwargs["sample"]["nmax"]),
    ]

    # Go!
    color = lambda i, alpha: "{}{}".format(
        colors.to_hex("C{}".format(i)),
        ("0" + hex(int(alpha * 256)).split("0x")[-1])[-2:],
    )
    fig = None
    cum_samples = np.empty((0, 5))
    for k in tqdm(
        range(len(mean)), disable=bool(int(os.getenv("NOTQDM", "0")))
    ):

        # Plot the 2d hist
        fig = corner(
            samples[k][:nsamp, :5],
            fig=fig,
            labels=labels,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            no_fill_contours=True,
            color=color(k, 0.1 * batch_alpha),
            contourf_kwargs=dict(),
            contour_kwargs=dict(alpha=0),
            bins=20,
            hist_bin_factor=5,
            smooth=2.0,
            hist_kwargs=dict(alpha=0),
            levels=(1.0 - np.exp(-0.5 * np.array([1.0]) ** 2)),
            range=ranges,
        )

        # Plot the 1d hist
        if k == len(mean) - 1:
            truths_ = truths
        else:
            truths_ = None
        fig = corner(
            samples[k][:nsamp, :5],
            fig=fig,
            labels=labels,
            plot_datapoints=False,
            plot_density=False,
            plot_contours=False,
            fill_contours=False,
            no_fill_contours=True,
            color=color(k, batch_alpha),
            bins=500,
            smooth1d=10.0,
            hist_kwargs=dict(alpha=0.5 * batch_alpha),
            range=ranges,
            truths=truths_,
            truth_color="#ff7f0e",
        )

        # Running list
        cum_samples = np.vstack((cum_samples, samples[k][:nsamp, :5]))

    # Plot the cumulative posterior
    np.random.shuffle(cum_samples)
    fig = corner(
        cum_samples,
        fig=fig,
        labels=labels,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        color="k",
        contourf_kwargs=dict(),
        contour_kwargs=dict(linewidths=1),
        bins=100,
        hist_bin_factor=5,
        smooth=1.0,
        hist_kwargs=dict(alpha=0),
        levels=(1.0 - np.exp(-0.5 * np.array([1.0]) ** 2)),
        range=ranges,
    )
    fig = corner(
        cum_samples[:nsamp],
        fig=fig,
        labels=labels,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        no_fill_contours=True,
        color="k",
        bins=500,
        smooth1d=10.0,
        hist_kwargs=dict(lw=1),
        range=ranges,
    )

    # Fix the axis limits
    ax = np.array(fig.axes).reshape(5, 5)
    for k in range(5):
        axis = ax[k, k]
        ymax = np.max([line._y.max() for line in axis.lines])
        axis.set_ylim(0, 1.1 * ymax)
        for axis in ax[:, k]:
            axis.set_xlim(*span[k])
        for axis in ax[k, :k]:
            axis.set_ylim(*span[k])

    # We're done
    fig.savefig(
        os.path.join(path, "calibration_corner.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Get the inclination posterior means and covariances (if present)
    inc_files = glob.glob(os.path.join(path, "*", "inclinations.npz"))
    if len(inc_files):

        data_files = [
            file.replace("inclinations", "data") for file in inc_files
        ]
        x = np.linspace(-90, 90, 1000)
        deltas = []

        # Compute the "posterior error" histogram
        for k in tqdm(
            range(len(inc_files)), disable=bool(int(os.getenv("NOTQDM", "0")))
        ):
            data = np.load(data_files[k])
            results = np.load(inc_files[k])
            truths = data["incs"]
            inc = results["inc"]
            lp = results["lp"]
            nlc = len(truths)
            nsamples = lp.shape[1]
            for n in range(nlc):
                for j in range(nsamples):
                    pdf = np.exp(lp[n, j] - np.max(lp[n, j]))
                    pdf /= np.trapz(pdf)
                    mean = np.trapz(inc * pdf)
                    var = np.trapz(inc ** 2 * pdf) - mean ** 2
                    deltas.append((mean - truths[n]) / np.sqrt(var))

        # Plot it
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.hist(deltas, bins=50, range=(-5, 5), density=True, label="measured")
        ax.hist(
            Normal.rvs(size=len(deltas)),
            bins=50,
            range=(-5, 5),
            histtype="step",
            color="C1",
            density=True,
            label=r"$\mathcal{N}(0, 1)$",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlabel("inclination posterior error")
        ax.set_ylabel("density")
        fig.savefig(
            os.path.join(path, "inclinations.pdf"), bbox_inches="tight"
        )
