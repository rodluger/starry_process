from .generate import generate
from .defaults import update_with_defaults
from .log_prob import get_log_prob
import numpy as np
import dynesty


def sample(data, clobber=False, **kwargs):

    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    gen_kwargs = kwargs["generate"]
    normalized = gen_kwargs["normalized"]
    period = gen_kwargs["period"]
    sample_kwargs = kwargs["sample"]
    ydeg = sample_kwargs["ydeg"]
    rmin = sample_kwargs["rmin"]
    rmax = sample_kwargs["rmax"]
    amin = sample_kwargs["amin"]
    amax = sample_kwargs["amax"]
    bmin = sample_kwargs["bmin"]
    bmax = sample_kwargs["bmax"]
    cmin = sample_kwargs["cmin"]
    cmax = sample_kwargs["cmax"]
    nmin = sample_kwargs["nmin"]
    nmax = sample_kwargs["nmax"]
    sampler = sample_kwargs["sampler"]
    sampler_kwargs = sample_kwargs["sampler_kwargs"]
    run_nested_kwargs = sample_kwargs["run_nested_kwargs"]
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
        marginalize_over_inclination=True,
    )

    # Prior transform
    def ptform(u):
        x = np.zeros_like(u)
        x[0] = rmin + u[0] * (rmax - rmin)
        x[1] = amin + u[1] * (amax - amin)
        x[2] = bmin + u[2] * (bmax - bmin)
        x[3] = cmin + u[3] * (cmax - cmin)
        x[4] = nmin + u[4] * (nmax - nmin)
        return x

    # Nested sampling
    ndim = 5
    if sampler == "NestedSampler":
        sampler = dynesty.NestedSampler(
            lambda x: log_prob(*x), ptform, ndim, **sampler_kwargs
        )
    elif sampler == "DynamicNestedSampler":
        sampler = dynesty.DynamicNestedSampler(
            lambda x: log_prob(*x), ptform, ndim, **sampler_kwargs
        )
    else:
        raise ValueError("Invalid sampler.")
    sampler.run_nested(**run_nested_kwargs)
    results = sampler.results
    return results
