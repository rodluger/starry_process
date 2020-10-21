from .defaults import update_with_defaults
from .. import StarryProcess
from ..optimize import NAdam
from ..math import cho_factor, cho_solve
import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse


def optimize(data, **kwargs):

    # Get kwargs
    kwargs = update_with_defaults(**kwargs)
    optim_kwargs = kwargs["optimize"]
    gen_kwargs = kwargs["generate"]
    normalized = gen_kwargs["normalized"]
    niter = optim_kwargs["niter"]
    ntries = optim_kwargs["ntries"]
    seed = optim_kwargs["seed"]
    nadam_kwargs = dict(lr=optim_kwargs["lr"])
    min_radius = optim_kwargs["min_radius"]
    max_radius = optim_kwargs["max_radius"]
    min_spots = optim_kwargs["min_spots"]
    max_spots = optim_kwargs["max_spots"]
    apply_jac = optim_kwargs["apply_jac"]
    ydeg = optim_kwargs["ydeg"]
    baseline_var = optim_kwargs["baseline_var"]

    # Get the data
    t = data["t"]
    flux = data["flux"]
    ferr = data["ferr"]
    p = data["period"]
    nlc = len(flux)

    # Vars (w/ test values)
    np.random.seed(seed)
    r = theano.shared(20.0)
    a = theano.shared(0.5)
    b = theano.shared(0.5)
    c = theano.shared(0.1)
    n = theano.shared(1.0)
    theano_vars = [r, a, b, c, n]

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
    loss = -(log_like + jac)

    # Set up the optimizer
    upd = NAdam(loss, theano_vars, **nadam_kwargs)
    train = theano.function([], theano_vars + [loss], updates=upd)

    # Initialize to the guess values
    r.set_value(np.random.uniform(min_radius, max_radius))
    a.set_value(np.random.random())
    b.set_value(np.random.random())
    c.set_value(np.random.random())
    n.set_value(np.random.uniform(min_spots, max_spots))

    # Iterate
    loss_val = np.zeros((ntries, niter))
    best_loss = np.inf
    best_params = []
    iterator_i = tqdm(np.arange(ntries))
    for i in iterator_i:
        iterator_j = tqdm(np.arange(niter))
        for j in iterator_j:
            *params, loss_val[i, j] = train()
            iterator_k.set_postfix({"loss": "{:.1f}".format(best_loss)})
            if loss_val[i, j] < best_loss:
                best_loss = loss_val[i, j]
                best_params = params

    return loss_val, best_params
