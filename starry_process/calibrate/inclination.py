from .defaults import update_with_defaults
from .log_prob import get_log_prob
import numpy as np
from tqdm.auto import tqdm
from dynesty import utils as dyfunc
import os


def compute_inclination_pdf(data, results, **kwargs):
    """

    """
    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    sample_kwargs = kwargs["sample"]
    gen_kwargs = kwargs["generate"]
    plot_kwargs = kwargs["plot"]
    ninc_pts = plot_kwargs["ninc_pts"]
    ninc_samples = plot_kwargs["ninc_samples"]
    ydeg = sample_kwargs["ydeg"]
    apply_jac = sample_kwargs["apply_jac"]
    normalized = gen_kwargs["normalized"]

    # Get the data
    t = data["t"]
    ferr = data["ferr"]
    period = data["period"]
    flux = data["flux"]
    nlc = len(flux)

    # Array of inclinations & log prob for each light curve
    inc = np.linspace(0, 90, ninc_pts)
    lp = np.empty((nlc, ninc_samples, ninc_pts))

    # Compile the likelihood function for a given inclination
    if sample_kwargs["fit_bm"]:
        baseline_mean = None
    else:
        baseline_mean = sample_kwargs["bm"]
    if sample_kwargs["fit_blv"]:
        baseline_log_var = None
    else:
        baseline_log_var = sample_kwargs["blv"]
    log_prob = get_log_prob(
        t,
        flux=None,
        ferr=ferr,
        p=period,
        ydeg=ydeg,
        baseline_mean=baseline_mean,
        baseline_log_var=baseline_log_var,
        apply_jac=apply_jac,
        normalized=normalized,
        marginalize_over_inclination=False,
    )

    # Resample posterior samples to equal weight
    samples = np.array(results.samples)
    try:
        weights = np.exp(results["logwt"] - results["logz"][-1])
    except:
        weights = results["weights"]
    samples = dyfunc.resample_equal(samples, weights)

    # Compute the posteriors
    for n in tqdm(range(nlc), disable=bool(int(os.getenv("NOTQDM", "0")))):
        for j in range(ninc_samples):
            idx = np.random.randint(len(samples))
            lp[n, j] = np.array(
                [
                    log_prob(flux[n].reshape(1, -1), *samples[idx], i)
                    for i in inc
                ]
            )

    return dict(inc=inc, lp=lp)

