from .. import StarryProcess
from ..math import cho_factor, cho_solve
from ..compat import theano, tt, ifelse
import numpy as np


def get_log_prob(
    t,
    flux=None,
    ferr=1.0e-3,
    p=1.0,
    ydeg=15,
    baseline_log_var=0.0,
    baseline_mean=0.0,
    apply_jac=True,
    normalized=True,
    marginalize_over_inclination=True,
    u=[0.0, 0.0],
):
    # Dimensions
    K = len(t)

    # Set up the model
    r = tt.dscalar()
    a = tt.dscalar()
    b = tt.dscalar()
    c = tt.dscalar()
    n = tt.dscalar()
    i = tt.dscalar()
    m = tt.dscalar()
    v = tt.dscalar()
    if flux is None:
        free_flux = True
        flux = tt.dmatrix()
    else:
        free_flux = False
    sp = StarryProcess(
        ydeg=ydeg,
        r=r,
        a=a,
        b=b,
        c=c,
        n=n,
        normalized=normalized,
        marginalize_over_inclination=marginalize_over_inclination,
        covpts=len(t) - 1,
    )

    # Get # of light curves in batch
    flux = tt.as_tensor_variable(flux)
    nlc = tt.shape(flux)[0]

    # Compute the mean and covariance of the process
    gp_mean = sp.mean(t, p=p, i=i, u=u)
    gp_cov = sp.cov(t, p=p, i=i, u=u)

    # Residual matrix
    R = tt.transpose(flux) - tt.reshape(gp_mean, (-1, 1))
    if baseline_mean is None:
        # Tensor variable
        R -= m
    else:
        # Fixed
        R -= baseline_mean

    # Observational error
    gp_cov += ferr ** 2 * tt.eye(K)

    # Marginalize over the baseline
    if baseline_log_var is None:
        # Tensor variable
        gp_cov += 10 ** v
    else:
        # Fixed
        gp_cov += 10 ** baseline_log_var

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
        log_prob = log_like + sp.log_jac()
    else:
        log_prob = log_like

    # Free variables
    theano_vars = [r, a, b, c, n]
    if baseline_mean is None:
        theano_vars += [m]
    if baseline_log_var is None:
        theano_vars += [v]
    if free_flux:
        theano_vars = [flux] + theano_vars
    if not marginalize_over_inclination:
        theano_vars = theano_vars + [i]

    # Compile & return
    log_prob = theano.function(theano_vars, log_prob)
    return log_prob
