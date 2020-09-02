from starry_process import StarryProcess
import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet


def get_functions():
    def _lnlike(
        alpha_s,
        beta_s,
        alpha_l,
        beta_l,
        mu_c,
        sigma_c,
        period,
        inc,
        t,
        flux,
        data_cov,
    ):
        gp = StarryProcess(
            alpha_s=alpha_s,
            beta_s=beta_s,
            mu_c=mu_c,
            sigma_c=sigma_c,
            alpha_l=alpha_l,
            beta_l=beta_l,
            period=period,
            inc=inc,
        )
        return gp.log_likelihood(t, flux, data_cov)

    def _sample(
        alpha_s, beta_s, alpha_l, beta_l, mu_c, sigma_c, period, inc, t,
    ):
        gp = StarryProcess(
            alpha_s=alpha_s,
            beta_s=beta_s,
            mu_c=mu_c,
            sigma_c=sigma_c,
            alpha_l=alpha_l,
            beta_l=beta_l,
            period=period,
            inc=inc,
        )
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
    params = [10.0, 30.0, 10.0, 30.0, 0.75, 0.1, 1.0, 60.0]
    t = np.linspace(0, 3, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of beta_l
    ll = np.zeros(100)
    beta = np.linspace(0.01, 100, len(ll))
    for i in range(len(ll)):
        params[3] = beta[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    # A very simple test
    assert np.abs(beta[np.argmax(ll)] - 30.0) < 5.0
