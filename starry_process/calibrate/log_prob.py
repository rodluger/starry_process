from .. import StarryProcess
from ..math import cho_factor, cho_solve
import numpy as np
import theano
import theano.tensor as tt
from theano.ifelse import ifelse


def get_log_prob(
    t,
    flux,
    ferr,
    p,
    ydeg=15,
    baseline_var=1e-4,
    apply_jac=True,
    normalized=True,
):

    # Dimensions
    nlc = len(flux)
    K = len(t)

    # Set up the model
    r = tt.dscalar()
    a = tt.dscalar()
    b = tt.dscalar()
    c = tt.dscalar()
    n = tt.dscalar()
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

    # Compile it into a function
    log_prob = theano.function([r, a, b, c, n], log_like + jac)

    return log_prob
