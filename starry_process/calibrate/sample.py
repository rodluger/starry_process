from .generate import generate
from .defaults import update_with_defaults
from .log_prob import get_log_prob
import numpy as np
import dynesty


def sample(data, clobber=False, **kwargs):

    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    name = kwargs["name"]
    gen_kwargs = kwargs["generate"]
    normalized = gen_kwargs["normalized"]
    period = gen_kwargs["period"]
    sample_kwargs = kwargs["sample"]
    ydeg = sample_kwargs["ydeg"]
    max_radius = sample_kwargs["max_radius"]
    min_radius = sample_kwargs["min_radius"]
    max_spots = sample_kwargs["max_spots"]
    min_spots = sample_kwargs["min_spots"]
    maxiter = sample_kwargs["maxiter"]
    maxiter_batch = sample_kwargs["maxiter_batch"]
    use_stop = sample_kwargs["use_stop"]
    baseline_var = sample_kwargs["baseline_var"]
    apply_jac = sample_kwargs["apply_jac"]

    # Get the log prob function for the dataset
    t = data["t"]
    flux = data["flux"]
    ferr = data["ferr"]
    log_prob = get_log_prob(
        t,
        flux,
        ferr,
        period,
        ydeg=ydeg,
        baseline_var=baseline_var,
        apply_jac=apply_jac,
        normalized=normalized,
    )

    # Prior transform
    def ptform(u):
        x = np.zeros_like(u)
        x[0] = min_radius + u[0] * (max_radius - min_radius)  # r
        x[1] = u[1]  # a
        x[2] = u[2]  # b
        x[3] = u[3]  # c
        x[4] = min_spots + u[4] * (max_spots - min_spots)  # n
        return x

    # Dynamic nested sampling
    ndim = 5
    sampler = dynesty.DynamicNestedSampler(
        lambda x: log_prob(*x), ptform, ndim
    )
    sampler.run_nested(
        maxiter=maxiter,
        wt_kwargs={"pfrac": 1.0},
        use_stop=use_stop,
        maxiter_batch=maxiter_batch,
    )
    results = sampler.results
    return results
