from starry_process import StarryProcess
from starry_process.defaults import defaults
import theano
import theano.tensor as tt
import timeit
import numpy as np
import pytest


@pytest.mark.xfail()
def test_timing(ydeg=15, npts=1000):

    # Free parameters
    sa = tt.dscalar()
    sb = tt.dscalar()
    la = tt.dscalar()
    lb = tt.dscalar()
    ca = tt.dscalar()
    cb = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()

    # Compute the mean and covariance
    gp = StarryProcess(
        sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, period=period, inc=inc,
    )
    mu = gp.mean(t)
    cov = gp.cov(t)

    # Compile the function
    get_mu_and_cov = theano.function(
        [sa, sb, la, lb, ca, cb, period, inc, t,], [mu, cov],
    )

    # Time it!
    number = 100
    t = np.linspace(0, 1, npts)
    time = (
        timeit.timeit(
            lambda: get_mu_and_cov(
                defaults["sa"],
                defaults["sb"],
                defaults["la"],
                defaults["lb"],
                defaults["ca"],
                defaults["cb"],
                defaults["period"],
                defaults["inc"],
                t,
            ),
            number=number,
        )
        / number
    )

    print("time elapsed: {:.4f} s".format(time))
    assert time < 0.1, "too slow! ({:.4f} s)".format(time)


def test_profile(ydeg=15, npts=1000):

    # Free parameters
    sa = tt.dscalar()
    sb = tt.dscalar()
    la = tt.dscalar()
    lb = tt.dscalar()
    ca = tt.dscalar()
    cb = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()
    flux = tt.dvector()
    data_cov = tt.dscalar()

    # Compute the mean and covariance
    gp = StarryProcess(
        sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, period=period, inc=inc
    )

    # Compile the function
    log_likelihood = theano.function(
        [sa, sb, la, lb, ca, cb, period, inc, t, flux, data_cov,],
        [gp.log_likelihood(t, flux, data_cov)],
        profile=True,
    )

    # Run it
    t = np.linspace(0, 1, npts)
    flux = np.random.randn(npts)
    data_cov = 1.0
    ll = log_likelihood(
        defaults["sa"],
        defaults["sb"],
        defaults["la"],
        defaults["lb"],
        defaults["ca"],
        defaults["cb"],
        defaults["period"],
        defaults["inc"],
        t,
        flux,
        data_cov,
    )

    # Log the summary
    print(log_likelihood.profile.summary())


if __name__ == "__main__":
    test_profile()
