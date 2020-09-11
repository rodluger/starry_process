from starry_process import StarryProcess
from starry_process.defaults import defaults
import theano
import theano.tensor as tt
import timeit
import numpy as np
import pytest
import warnings


@pytest.mark.parametrize(
    "gradient,profile",
    [[False, False], [False, True], [True, False], [True, True]],
)
def test_profile(gradient, profile, ydeg=15, npts=1000):

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
    if gradient:
        g = lambda f, x: tt.grad(f, x)
    else:
        g = lambda f, x: f
    func = theano.function(
        [sa, sb, la, lb, ca, cb, period, inc, t, flux, data_cov,],
        [g(gp.log_likelihood(t, flux, data_cov), la)],
        profile=profile,
    )

    # Run it
    t = np.linspace(0, 1, npts)
    flux = np.random.randn(npts)
    data_cov = 1.0

    run = lambda: func(
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

    if profile:

        # Profile the full function
        run()
        print(func.profile.summary())

    else:

        # Time the execution
        number = 100
        time = timeit.timeit(run, number=number,) / number
        print("time elapsed: {:.4f} s".format(time))
        if (gradient and time > 0.2) or (not gradient and time > 0.1):
            warnings.warn("too slow! ({:.4f} s)".format(time))
