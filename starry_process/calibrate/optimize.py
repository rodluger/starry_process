from .defaults import update_with_defaults
from .. import StarryProcess
import pymc3 as pm
from pymc3_ext import optim as op
from ..math import cho_factor, cho_solve
import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from tqdm.auto import tqdm


def optimize(data, **kwargs):

    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    optim_kwargs = kwargs["optimize"]
    gen_kwargs = kwargs["generate"]
    normalized = gen_kwargs["normalized"]
    niter = optim_kwargs["niter"]
    ntries = optim_kwargs["ntries"]
    seed = optim_kwargs["seed"]
    adam_kwargs = dict(lr=optim_kwargs["lr"])
    min_radius = optim_kwargs["min_radius"]
    max_radius = optim_kwargs["max_radius"]
    min_spots = optim_kwargs["min_spots"]
    max_spots = optim_kwargs["max_spots"]
    apply_jac = optim_kwargs["apply_jac"]
    ydeg = optim_kwargs["ydeg"]
    baseline_var = optim_kwargs["baseline_var"]

    def get_guesses():
        r0 = min_radius + np.random.random() * (max_radius - min_radius)
        a0 = np.random.random()
        b0 = np.random.random()
        c0 = np.random.random()
        n0 = min_spots + np.random.random() * (max_spots - min_spots)
        return r0, a0, b0, c0, n0

    # Get the data
    t = data["t"]
    flux = data["flux"]
    ferr = data["ferr"]
    p = data["period"]
    nlc = len(flux)

    # Init
    np.random.seed(seed)
    r0, a0, b0, c0, n0 = get_guesses()

    with pm.Model() as model:

        # Vars
        r = pm.Uniform("r", lower=min_radius, upper=max_radius, testval=r0)
        a = pm.Uniform("a", lower=0, upper=1, testval=a0)
        b = pm.Uniform("b", lower=0, upper=1, testval=b0)
        c = pm.Uniform("c", lower=0, upper=1, testval=c0)
        n = pm.Uniform("n", lower=min_spots, upper=max_spots, testval=n0)

        # Compute the loss
        K = len(t)
        sp = StarryProcess(
            ydeg=ydeg,
            r=r,
            a=a,
            b=b,
            c=c,
            n=n,
            marginalize_over_inclination=True,
            covpts=len(t) - 1,
        )

        # Compute the mean and covariance of the process
        gp_mean = sp.mean(t, p=p)
        gp_cov = sp.cov(t, p=p)

        if normalized:

            # Assume the data is normalized to zero mean.
            # We need to scale our covariance accordingly
            gp_cov /= (1 + gp_mean) ** 2
            R = tt.as_tensor_variable(flux.T)

        else:

            # Assume we can measure the true baseline,
            # which is just the mean of the GP
            R = tt.as_tensor_variable(flux.T) - tt.reshape(gp_mean, (-1, 1))

        # Observational error
        gp_cov += ferr ** 2 * tt.eye(K)

        # Marginalize over the baseline
        gp_cov += baseline_var

        # Compute the batched likelihood
        cho_gp_cov = cho_factor(gp_cov)
        CInvR = cho_solve(cho_gp_cov, R)
        log_like = -0.5 * tt.sum(
            tt.batched_dot(tt.transpose(R), tt.transpose(CInvR))
        )
        log_like -= nlc * tt.sum(tt.log(tt.diag(cho_gp_cov)))
        log_like -= 0.5 * nlc * K * tt.log(2 * np.pi)
        log_like = ifelse(tt.isnan(log_like), -np.inf, log_like)

        # Latitude jacobian
        if apply_jac:
            jac = sp.log_jac()
        else:
            jac = 0.0

        # Compute the loss
        pm.Potential("loss", -(log_like + jac))

        # Iterate
        loss = np.zeros((ntries, niter))
        best_loss = np.inf
        best_params = []
        optimizer = op.Adam(**adam_kwargs)
        iterator_i = tqdm(np.arange(ntries))
        for i in iterator_i:
            iterator_j = tqdm(
                op.optimize_iterator(optimizer, niter, vars=[r, a, b, c, n])
            )
            j = 0
            for obj, point in iterator_j:
                loss[i, j] = obj
                if loss[i, j] < best_loss:
                    best_loss = loss[i, j]
                    best_params = point
                iterator_j.set_postfix({"loss": "{:.1f}".format(best_loss)})
                j += 1

    return loss, best_params
