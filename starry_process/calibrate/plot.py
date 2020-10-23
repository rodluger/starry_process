from .defaults import update_with_defaults
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from starry_process import StarryProcess
import theano
import theano.tensor as tt
import numpy as np
from scipy.stats import norm as Normal
import dynesty.plotting as dyplot


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
    samples = results.samples

    _draw_pdf = lambda x, a, b: StarryProcess(a=a, b=b).latitude.pdf(x)
    _x = tt.dvector()
    _a = tt.dscalar()
    _b = tt.dscalar()
    draw_pdf = theano.function([_x, _a, _b], _draw_pdf(_x, _a, _b),)
    x = np.linspace(-89.9, 89.9, nlat_pts)
    pdf_true = 0.5 * (
        Normal.pdf(x, mu_true, sigma_true)
        + Normal.pdf(x, -mu_true, sigma_true)
    )
    pdf = np.empty((nlat_samples, nlat_pts))
    for k in range(nlat_samples):
        idx = np.random.randint(len(samples))
        pdf[k] = draw_pdf(x, samples[idx, 1], samples[idx, 2])
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


def plot_corner(results, **kwargs):
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

    fig, _ = dyplot.cornerplot(results, truths=truths, labels=labels)
    return fig
