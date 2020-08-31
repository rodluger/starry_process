from starry_process import StarryProcess
import theano
import theano.tensor as tt
import timeit
import numpy as np
import pytest


@pytest.mark.xfail()
def test_timing(ydeg=15, npts=1000):

    # Free parameters
    size_alpha = tt.dscalar()
    size_beta = tt.dscalar()
    latitude_alpha = tt.dscalar()
    latitude_beta = tt.dscalar()
    contrast_mu = tt.dscalar()
    contrast_sigma = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()

    # Compute the mean and covariance
    gp = StarryProcess(ydeg)
    gp.size.set_params(size_alpha, size_beta)
    gp.latitude.set_params(latitude_alpha, latitude_beta)
    gp.contrast.set_params(contrast_mu, contrast_sigma)
    gp.design.set_params(period, inc)
    mu = gp.mean(t)
    cov = gp.cov(t)

    # Compile the function
    get_mu_and_cov = theano.function(
        [
            size_alpha,
            size_beta,
            latitude_alpha,
            latitude_beta,
            contrast_mu,
            contrast_sigma,
            period,
            inc,
            t,
        ],
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
    assert time < 0.1, "too slow!"
