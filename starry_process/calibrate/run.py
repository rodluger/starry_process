from .defaults import update_with_defaults
from starry_process import calibrate
from starry_process.latitude import beta2gauss
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
    plot_inclination=False,
    **kwargs
):
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the kwargs
    if clobber or not os.path.exists(os.path.join(path, "kwargs.json")):
        json.dump(
            update_with_defaults(**kwargs),
            open(os.path.join(path, "kwargs.json"), "w"),
        )
    else:
        kwargs = json.load(open(os.path.join(path, "kwargs.json"), "r"))

    # Generate
    if clobber or not os.path.exists(os.path.join(path, "data.npz")):
        data = calibrate.generate(**kwargs)
        np.savez(os.path.join(path, "data.npz"), **data)
    else:
        data = np.load(os.path.join(path, "data.npz"))

    # Plot the data
    if plot_all or plot_data:
        fig = calibrate.plot_data(data, **kwargs)
        fig.savefig(os.path.join(path, "data.pdf"), bbox_inches="tight")

    # Sample
    if clobber or not os.path.exists(os.path.join(path, "results.pkl")):
        results = calibrate.sample(data, **kwargs)
        pickle.dump(results, open(os.path.join(path, "results.pkl"), "wb"))
    else:
        results = pickle.load(open(os.path.join(path, "results.pkl"), "rb"))

    # Transform latitude params and store posterior mean and cov
    samples = np.array(results.samples)
    samples[:, 1], samples[:, 2] = beta2gauss(samples[:, 1], samples[:, 2])
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    np.savez(os.path.join(path, "mean_and_cov.npz"), mean=mean, cov=cov)

    # Plot the results
    if plot_all or plot_latitude_pdf:
        fig = calibrate.plot_latitude_pdf(results, **kwargs)
        fig.savefig(os.path.join(path, "latitude.pdf"), bbox_inches="tight")

    if plot_all or plot_trace:
        fig = calibrate.plot_trace(results, **kwargs)
        fig.savefig(os.path.join(path, "trace.pdf"), bbox_inches="tight")

    if plot_all or plot_corner:
        fig = calibrate.plot_corner(results, transform_beta=False, **kwargs)
        fig.savefig(os.path.join(path, "corner.pdf"), bbox_inches="tight")

    if plot_all or plot_corner_transformed:
        fig = calibrate.plot_corner(results, transform_beta=True, **kwargs)
        fig.savefig(
            os.path.join(path, "corner_transformed.pdf"), bbox_inches="tight"
        )

    if plot_all or plot_inclination:
        fig = calibrate.plot_inclination_pdf(data, results, **kwargs)
        fig.savefig(os.path.join(path, "inclination.pdf"), bbox_inches="tight")