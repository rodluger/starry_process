from starry_process import StarryProcess
from starry_process.defaults import defaults
from starry_process.compat import theano, tt
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
    r = tt.dscalar()
    a = tt.dscalar()
    b = tt.dscalar()
    c = tt.dscalar()
    n = tt.dscalar()
    i = tt.dscalar()
    p = tt.dscalar()
    t = tt.dvector()
    flux = tt.dvector()
    data_cov = tt.dscalar()

    # Compute the mean and covariance
    gp = StarryProcess(
        r=r, a=a, b=b, c=c, n=n, marginalize_over_inclination=False
    )

    # Compile the function
    if gradient:
        g = lambda f, x: tt.grad(f, x)
    else:
        g = lambda f, x: f
    func = theano.function(
        [r, a, b, c, n, i, p, t, flux, data_cov],
        [
            g(gp.log_likelihood(t, flux, data_cov, i=i, p=p), a)
        ],  # wrt a for definiteness
        profile=profile,
    )

    # Run it
    t = np.linspace(0, 1, npts)
    flux = np.random.randn(npts)
    data_cov = 1.0

    run = lambda: func(
        defaults["r"],
        defaults["a"],
        defaults["b"],
        defaults["c"],
        defaults["n"],
        defaults["i"],
        defaults["p"],
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
        time = timeit.timeit(run, number=number) / number
        print("time elapsed: {:.4f} s".format(time))
        if (gradient and time > 0.2) or (not gradient and time > 0.1):
            warnings.warn("too slow! ({:.4f} s)".format(time))


@pytest.mark.parametrize(
    "gradient,profile",
    [[False, False], [False, True], [True, False], [True, True]],
)
def test_profile_marg(gradient, profile, ydeg=15, npts=1000):

    # Free parameters
    r = tt.dscalar()
    a = tt.dscalar()
    b = tt.dscalar()
    c = tt.dscalar()
    n = tt.dscalar()
    p = tt.dscalar()
    t = tt.dvector()
    flux = tt.dvector()
    data_cov = tt.dscalar()

    # Compute the mean and covariance
    gp = StarryProcess(
        r=r, a=a, b=b, c=c, n=n, marginalize_over_inclination=True
    )

    # Compile the function
    if gradient:
        g = lambda f, x: tt.grad(f, x)
    else:
        g = lambda f, x: f
    func = theano.function(
        [r, a, b, c, n, p, t, flux, data_cov],
        [
            g(gp.log_likelihood(t, flux, data_cov, p=p), a)
        ],  # wrt a for definiteness
        profile=profile,
    )

    # Run it
    t = np.linspace(0, 1, npts)
    flux = np.random.randn(npts)
    data_cov = 1.0

    run = lambda: func(
        defaults["r"],
        defaults["a"],
        defaults["b"],
        defaults["c"],
        defaults["n"],
        defaults["p"],
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
        time = timeit.timeit(run, number=number) / number
        print("time elapsed: {:.4f} s".format(time))
        if (gradient and time > 0.2) or (not gradient and time > 0.1):
            warnings.warn("too slow! ({:.4f} s)".format(time))
