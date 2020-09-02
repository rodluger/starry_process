from starry_process import StarryProcess
import theano
import theano.tensor as tt
import timeit
import numpy as np
import pytest


@pytest.mark.xfail()
def test_timing(ydeg=15, npts=1000):

    # Free parameters
    alpha_s = tt.dscalar()
    beta_s = tt.dscalar()
    alpha_l = tt.dscalar()
    beta_l = tt.dscalar()
    mu_c = tt.dscalar()
    sigma_c = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()

    # Compute the mean and covariance
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
    mu = gp.mean(t)
    cov = gp.cov(t)

    # Compile the function
    get_mu_and_cov = theano.function(
        [alpha_s, beta_s, alpha_l, beta_l, mu_c, sigma_c, period, inc, t,],
        [mu, cov],
    )

    # Time it!
    number = 100
    t = np.linspace(0, 1, npts)
    time = (
        timeit.timeit(
            lambda: get_mu_and_cov(10.0, 30.0, 10.0, 50.0, 0.5, 0.1, 3, 65, t),
            number=number,
        )
        / number
    )

    print("time elapsed: {:.4f} s".format(time))
    assert time < 0.1, "too slow! ({:.4f} s)".format(time)


def test_profile(ydeg=15, npts=1000):

    # Free parameters
    alpha_s = tt.dscalar()
    beta_s = tt.dscalar()
    alpha_l = tt.dscalar()
    beta_l = tt.dscalar()
    mu_c = tt.dscalar()
    sigma_c = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()
    flux = tt.dvector()
    data_cov = tt.dscalar()

    # Compute the mean and covariance
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

    # Compile the function
    log_likelihood = theano.function(
        [
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
        ],
        [gp.log_likelihood(t, flux, data_cov)],
        profile=True,
    )

    # Run it
    t = np.linspace(0, 1, npts)
    flux = np.random.randn(npts)
    data_cov = 1.0
    ll = log_likelihood(
        10.0, 30.0, 10.0, 50.0, 0.5, 0.1, 3, 65, t, flux, data_cov
    )

    # Log the summary
    print(log_likelihood.profile.summary())
