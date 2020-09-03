from starry_process import StarryProcess
import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet


def get_functions():
    def _lnlike(
        mu_s,
        sigma_s,
        mu_l,
        sigma_l,
        mu_c,
        sigma_c,
        period,
        inc,
        t,
        flux,
        data_cov,
    ):
        gp = StarryProcess(
            mu_s=mu_s,
            sigma_s=sigma_s,
            mu_c=mu_c,
            sigma_c=sigma_c,
            mu_l=mu_l,
            sigma_l=sigma_l,
            period=period,
            inc=inc,
        )
        return gp.log_likelihood(t, flux, data_cov)

    def _sample(
        mu_s, sigma_s, mu_l, sigma_l, mu_c, sigma_c, period, inc, t,
    ):
        gp = StarryProcess(
            mu_s=mu_s,
            sigma_s=sigma_s,
            mu_c=mu_c,
            sigma_c=sigma_c,
            mu_l=mu_l,
            sigma_l=sigma_l,
            period=period,
            inc=inc,
        )
        gp.random.seed(42)
        return tt.reshape(gp.sample(t), (-1,))

    # Likelihood func
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector(), tt.dvector(), tt.dscalar()]
    lnlike = theano.function(inputs, _lnlike(*inputs))

    # Draw a sample
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector()]
    sample = theano.function(inputs, _sample(*inputs))

    return lnlike, sample


def test_lnlike_array():

    # Get the functions
    lnlike, sample = get_functions()

    # Generate a dataset
    mu_s = 15.0
    sigma_s = 5.0
    mu_l = 10.0
    sigma_l = 5.0
    mu_c = 0.75
    sigma_c = 0.1
    period = 1.0
    inc = 60.0
    params = [
        mu_s,
        sigma_s,
        mu_l,
        sigma_l,
        mu_c,
        sigma_c,
        period,
        inc,
    ]
    t = np.linspace(0, 3, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    np.random.seed(42)
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of `mu_l`
    ll = np.zeros(100)
    mu_l_arr = np.linspace(0.0, 90.0, len(ll))
    for i in range(len(ll)):
        params[2] = mu_l_arr[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    # A *very* simple test
    assert np.abs(mu_l_arr[np.nanargmax(ll)] - mu_l) < 10.0
