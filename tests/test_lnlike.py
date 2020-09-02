from starry_process import StarryProcess
import numpy as np
import theano.tensor as tt
import theano


def get_functions():
    def _lnlike(
        size_alpha,
        size_beta,
        latitude_alpha,
        latitude_beta,
        contrast_mu,
        contrast_sigma,
        period,
        inc,
        t,
        flux,
        data_cov,
    ):
        sp = StarryProcess()
        sp.size.set_params(size_alpha, size_beta)
        sp.latitude.set_params(latitude_alpha, latitude_beta)
        sp.contrast.set_params(contrast_mu, contrast_sigma)
        sp.design.set_params(period, inc)
        return sp.log_likelihood(t, flux, data_cov)

    def _sample(
        size_alpha,
        size_beta,
        latitude_alpha,
        latitude_beta,
        contrast_mu,
        contrast_sigma,
        period,
        inc,
        t,
    ):
        sp = StarryProcess()
        sp.size.set_params(size_alpha, size_beta)
        sp.latitude.set_params(latitude_alpha, latitude_beta)
        sp.contrast.set_params(contrast_mu, contrast_sigma)
        sp.design.set_params(period, inc)
        return tt.reshape(sp.sample(t), (-1,))

    # Likelihood func
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector(), tt.dvector(), tt.dscalar()]
    lnlike = theano.function(inputs, _lnlike(*inputs))

    # Draw a sample
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector()]
    sample = theano.function(inputs, _sample(*inputs))

    return lnlike, sample


def test_lnlike():

    # Get the functions
    lnlike, sample = get_functions()

    # Generate a dataset
    params = [10.0, 30.0, 10.0, 30.0, 0.75, 0.1, 1.0, 60.0]
    t = np.linspace(0, 3, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of latitude_beta
    ll = np.zeros(100)
    beta = np.linspace(0.01, 100, len(ll))
    for i in range(len(ll)):
        params[3] = beta[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    # TODO: Test this more robustly
    assert np.abs(beta[np.argmax(ll)] - 30.0) < 5.0
