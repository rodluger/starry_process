from .defaults import update_with_defaults
from .log_prob import get_log_prob
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from starry_process import StarryProcess
import theano
import theano.tensor as tt
import numpy as np
from scipy.stats import norm as Normal
import dynesty.plotting as dyplot
from dynesty import utils as dyfunc
from corner import corner as _corner
from tqdm.auto import tqdm


def corner(*args, **kwargs):
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
        ax.set_ylabel(ax.get_ylabel(), fontsize=kwargs["corner_label_size"])
    for ax in axes[-1, :]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(ax.get_xlabel(), fontsize=kwargs["corner_label_size"])

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


def mode_and_std(alpha, beta):
    alpha = np.atleast_1d(alpha)
    beta = np.atleast_1d(beta)

    # Compute the mode
    term = (
        4 * alpha ** 2
        - 8 * alpha
        - 6 * beta
        + 4 * alpha * beta
        + beta ** 2
        + 5
    )
    mode = 2 * np.arctan(np.sqrt(2 * alpha + beta - 2 - np.sqrt(term)))

    # Compute the curvature at the mode
    # and convert it to a standard deviation
    term = (
        1
        - alpha
        + beta
        + (beta - 1) * np.cos(mode)
        + (alpha - 1) / np.cos(mode) ** 2
    )
    std = np.sin(mode) / np.sqrt(term)

    # Invalid if alpha, beta < 1
    mode[(alpha < 1) | (beta < 1)] = np.nan
    std[(alpha < 1) | (beta < 1)] = np.nan

    return mode * 180 / np.pi, std * 180 / np.pi


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


def plot_data(data, **kwargs):

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
    images = data["images"]

    # Plot the synthetic dataset
    nlc = len(flux)
    if nlc > 10:
        nrows = int(np.ceil(nlc / 10))
    else:
        nrows = 1
    wr = np.ones(min(nlc, 10))
    wr[-1] = 1.17
    gridspec = {"width_ratios": wr}
    fig, ax = plt.subplots(
        2 * nrows,
        min(nlc, 10),
        figsize=(min(nlc, 10) + 2, 2 * nrows),
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
    for k in range(nlc):
        im = axtop[k].imshow(
            images[k],
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
        axbot[k].plot(
            t, 1e3 * (flux[k] - np.median(flux[k])), "k.", alpha=0.3, ms=1
        )
        axbot[k].plot(t, 1e3 * (flux0[k] - np.median(flux0[k])), "C0-", lw=1)
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

    for k in range(nlc, len(axtop)):
        axtop[k].axis("off")
        axbot[k].axis("off")

    return fig


def plot_latitude_pdf(results, **kwargs):
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
    draw_pdf = theano.function([_x, _a, _b], _draw_pdf(_x, _a, _b),)
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
        ax.plot(x, pdf[k], "C0-", lw=1, alpha=0.05)
    ax.plot(x, pdf_true, "C1-", label="truth")
    ax.plot(x, np.nan * x, "C0-", label="samples")
    ax.legend(loc="upper right")
    ax.set_xlim(-90, 90)
    xticks = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:d}$^\circ$".format(xt) for xt in xticks])
    ax.set_xlabel("latitude")
    ax.set_ylabel("probability")
    mx = np.max(pdf_true)
    ax.set_ylim(-0.1 * mx, 2.0 * mx)
    return fig


def plot_trace(results, **kwargs):
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    gen_kwargs = kwargs["generate"]
    labels = ["r", "a", "b", "c", "n"]

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
    ]
    fig, _ = dyplot.traceplot(results, truths=truths, labels=labels)
    return fig


def plot_corner(results, transform_beta=False, **kwargs):
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    gen_kwargs = kwargs["generate"]
    sample_kwargs = kwargs["sample"]
    plot_kwargs = kwargs["plot"]
    use_corner = plot_kwargs["use_corner"]
    span = [
        (sample_kwargs["rmin"], sample_kwargs["rmax"]),
        (sample_kwargs["amin"], sample_kwargs["amax"]),
        (sample_kwargs["bmin"], sample_kwargs["bmax"]),
        (sample_kwargs["cmin"], sample_kwargs["cmax"]),
        (sample_kwargs["nmin"], sample_kwargs["nmax"]),
    ]
    labels = [r"$r$", r"$a$", r"$b$", r"$c$", r"$n$"]

    # Get truths
    sp = StarryProcess()
    try:
        a, b = sp.latitude._transform.transform(
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
    ]

    if use_corner:

        samples = np.array(results.samples)

        if transform_beta:

            # Transform from `a, b` to `mode, std`
            a = samples[:, 1]
            b = samples[:, 2]
            alpha, beta = sp.latitude._transform._ab_to_alphabeta(a, b)
            mode, std = mode_and_std(alpha, beta)
            samples[:, 1] = mode
            samples[:, 2] = std
            labels[1] = r"$\mu$"
            labels[2] = r"$\sigma$"
            if np.isfinite(gen_kwargs["latitude"]["sigma"]):
                truths[1] = gen_kwargs["latitude"]["mu"]
                truths[2] = gen_kwargs["latitude"]["sigma"]
            else:
                truths[1] = np.nan
                truths[2] = np.nan
            span[1] = (0, 90)
            span[2] = (0, 30)

        # Get sample weights
        try:
            weights = np.exp(results["logwt"] - results["logz"][-1])
        except:
            weights = results["weights"]

        fig = corner(
            samples,
            plot_datapoints=False,
            plot_density=False,
            truths=truths,
            labels=labels,
            range=span,
            fill_contours=True,
            weights=weights,
            smooth=2.0,
            smooth1d=2.0,
            bins=100,
            hist_kwargs=dict(lw=1),
            **plot_kwargs
        )

    else:

        if transform_beta:
            # TODO: issue a warning
            pass

        fig, _ = dyplot.cornerplot(
            results,
            truths=truths,
            labels=labels,
            span=span,
            truth_color="#4682b4",
        )

    return fig


def plot_inclination_pdf(data, results, **kwargs):

    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    sample_kwargs = kwargs["sample"]
    gen_kwargs = kwargs["generate"]
    plot_kwargs = kwargs["plot"]
    ninc_pts = plot_kwargs["ninc_pts"]
    ninc_samples = plot_kwargs["ninc_samples"]
    ninc_plots = plot_kwargs["ninc_plots"]
    ydeg = sample_kwargs["ydeg"]
    baseline_var = sample_kwargs["baseline_var"]
    apply_jac = sample_kwargs["apply_jac"]
    normalized = gen_kwargs["normalized"]

    # Get the data
    t = data["t"]
    flux = data["flux"]
    ferr = data["ferr"]
    period = data["period"]

    # Resample posterior samples to equal weight
    samples = np.array(results.samples)
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]
    samples = dyfunc.resample_equal(samples, weights)

    # Get the likelihood function for a given inclination
    log_prob = get_log_prob(
        t,
        flux=None,
        ferr=ferr,
        p=period,
        ydeg=ydeg,
        baseline_var=baseline_var,
        apply_jac=apply_jac,
        normalized=normalized,
        marginalize_over_inclination=False,
    )

    # Array of inclinations
    inc = np.linspace(0, 90, ninc_pts)

    # Compute the inclination pdf for various posterior samples
    fig, ax = plt.subplots(1, ninc_plots, figsize=(3 * ninc_plots, 2))
    for n in tqdm(range(ninc_plots)):
        for j in tqdm(range(ninc_samples)):
            idx = np.random.randint(len(samples))
            ll = np.array(
                [
                    log_prob(flux[n].reshape(1, -1), *samples[idx], i)
                    for i in inc
                ]
            )
            ax[n].plot(inc, np.exp(ll - ll.max()), "C0-", lw=1, alpha=0.25)
        ax[n].axvline(data["incs"][n], color="C1")
        ax[n].set_yticks([])
        ax[n].set_xlim(0, 90)
        ax[n].set_xticks([0, 30, 60, 90])
        ax[n].set_xlabel("inclination {}".format(n + 1))
    return fig
