from starry_process import StarryProcess
from starry_process.latitude import beta2gauss
import numpy as np
import emcee
from starry_process.compat import theano, tt
from scipy.stats import median_abs_deviation as mad


def test_jacobian():

    # Compile the Jacobian
    _a = tt.dscalar()
    _b = tt.dscalar()
    log_jac = theano.function([_a, _b], StarryProcess(a=_a, b=_b).log_jac())

    # Log probability
    def log_prob(p):
        if np.any(p < 0):
            return -np.inf
        elif np.any(p > 1):
            return -np.inf
        else:
            return log_jac(*p)

    # Run the sampler
    ndim, nwalkers, nsteps = 2, 50, 10000
    p0 = np.random.random(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps)

    # Transform to latitude params
    a, b = sampler.chain.T.reshape(2, -1)
    mu, sigma = beta2gauss(a, b)

    # Compute the 2d histogram
    m1, m2 = 0, 80
    s1, s2 = 0, 45
    hist, _, _ = np.histogram2d(mu, sigma, range=((m1, m2), (s1, s2)))
    hist /= np.max(hist)

    # Check that the variation is less than 10% across the domain
    std = 1.4826 * mad(hist.flatten())
    mean = np.mean(hist.flatten())
    assert std / mean < 0.1
