from . import plot, defaults
from . import generate, sample, inclination
from ..latitude import beta2gauss
from dynesty import utils as dyfunc
import pickle
import numpy as np
import os
import json


def run(
    path=".",
    clobber=False,
    plot_all=False,
    plot_data=True,
    plot_latitude_pdf=True,
    plot_trace=False,
    plot_corner=False,
    plot_corner_transformed=True,
    plot_inclination_pdf=True,
    ncols=10,
    clip=False,
    fail_on_kwargs_mismatch=True,
    **kwargs,
):
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the kwargs
    if clobber or not os.path.exists(os.path.join(path, "kwargs.json")):
        json.dump(
            defaults.update_with_defaults(**kwargs),
            open(os.path.join(path, "kwargs.json"), "w"),
        )
    else:
        input_kwargs = defaults.update_with_defaults(**kwargs)
        saved_kwargs = json.load(open(os.path.join(path, "kwargs.json"), "r"))
        if not (input_kwargs == saved_kwargs):
            if fail_on_kwargs_mismatch:
                raise ValueError(
                    "Input kwargs don't match saved kwargs for this run."
                )
        kwargs = saved_kwargs

    # Generate
    if clobber or not os.path.exists(os.path.join(path, "data.npz")):
        data = generate(**kwargs)
        np.savez(os.path.join(path, "data.npz"), **data)
    else:
        data = np.load(os.path.join(path, "data.npz"))

    # Plot the data
    if plot_all or plot_data:
        if clobber or not os.path.exists(os.path.join(path, "data.pdf")):
            fig = plot.plot_data(data, ncols=ncols, clip=clip, **kwargs)
            fig.savefig(
                os.path.join(path, "data.pdf"), bbox_inches="tight", dpi=300
            )

    # Sample
    if clobber or not os.path.exists(os.path.join(path, "results.pkl")):
        results = sample(data, **kwargs)
        pickle.dump(results, open(os.path.join(path, "results.pkl"), "wb"))
    else:
        results = pickle.load(open(os.path.join(path, "results.pkl"), "rb"))

    # Compute inclination pdf
    compute_inclination_pdf = defaults.update_with_defaults(**kwargs)[
        "sample"
    ]["compute_inclination_pdf"]
    if compute_inclination_pdf:
        if clobber or not os.path.exists(
            os.path.join(path, "inclinations.npz")
        ):
            inc_results = inclination.compute_inclination_pdf(
                data, results, **kwargs
            )
            np.savez(os.path.join(path, "inclinations.npz"), **inc_results)
        else:
            inc_results = np.load(os.path.join(path, "inclinations.npz"))
    else:
        inc_results = None

    # Transform latitude params and store posterior mean and cov
    if clobber or not os.path.exists(os.path.join(path, "mean_and_cov.npz")):
        samples = np.array(results.samples)
        samples[:, 1], samples[:, 2] = beta2gauss(samples[:, 1], samples[:, 2])
        try:
            weights = np.exp(results["logwt"] - results["logz"][-1])
        except:
            weights = results["weights"]
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        np.savez(os.path.join(path, "mean_and_cov.npz"), mean=mean, cov=cov)
    else:
        mean_and_cov = np.load(os.path.join(path, "mean_and_cov.npz"))
        mean = mean_and_cov["mean"]
        cov = mean_and_cov["cov"]

    # Plot the results
    if plot_all or plot_latitude_pdf:
        if clobber or not os.path.exists(os.path.join(path, "latitude.pdf")):
            fig = plot.plot_latitude_pdf(results, **kwargs)
            fig.savefig(
                os.path.join(path, "latitude.pdf"), bbox_inches="tight"
            )

    if plot_all or plot_trace:
        if clobber or not os.path.exists(os.path.join(path, "trace.pdf")):
            fig = plot.plot_trace(results, **kwargs)
            fig.savefig(os.path.join(path, "trace.pdf"), bbox_inches="tight")

    if plot_all or plot_corner:
        if clobber or not os.path.exists(os.path.join(path, "corner.pdf")):
            fig = plot.plot_corner(results, transform_beta=False, **kwargs)
            fig.savefig(os.path.join(path, "corner.pdf"), bbox_inches="tight")

    if plot_all or plot_corner_transformed:
        if clobber or not os.path.exists(
            os.path.join(path, "corner_transformed.pdf")
        ):
            fig = plot.plot_corner(results, transform_beta=True, **kwargs)
            fig.savefig(
                os.path.join(path, "corner_transformed.pdf"),
                bbox_inches="tight",
            )

    if (plot_all or plot_inclination_pdf) and (compute_inclination_pdf):
        if clobber or not os.path.exists(
            os.path.join(path, "inclination.pdf")
        ):
            fig = plot.plot_inclination_pdf(data, inc_results, **kwargs)
            fig.savefig(
                os.path.join(path, "inclination.pdf"), bbox_inches="tight"
            )
