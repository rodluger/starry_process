from starry_process import StarryProcess
from starry_process.defaults import defaults
import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet
import matplotlib.pyplot as plt
from theano.tests.unittest_tools import verify_grad
from theano.configparser import change_flags
import pytest
from tqdm import tqdm
from itertools import product


def get_functions(npts=100, marginalize_over_inclination=False):
    def _lnlike(
        r, a, b, c, n, i, p, t, flux, data_cov,
    ):
        gp = StarryProcess(
            r=r,
            a=a,
            b=b,
            c=c,
            n=n,
            marginalize_over_inclination=marginalize_over_inclination,
        )
        return gp.log_likelihood(t, flux, data_cov, p=p, i=i, npts=npts)

    def _sample(
        r, a, b, c, n, i, p, t,
    ):
        gp = StarryProcess(
            r=r,
            a=a,
            b=b,
            c=c,
            n=n,
            marginalize_over_inclination=marginalize_over_inclination,
        )
        gp.random.seed(42)
        return tt.reshape(gp.sample(t, p=p, i=i, npts=npts), (-1,))

    # Likelihood func
    inputs = [tt.dscalar() for n in range(7)]
    inputs += [tt.dvector(), tt.dvector(), tt.dscalar()]
    lnlike = theano.function(
        inputs, _lnlike(*inputs), on_unused_input="ignore"
    )

    # Draw a sample
    inputs = [tt.dscalar() for n in range(7)]
    inputs += [tt.dvector()]
    sample = theano.function(
        inputs, _sample(*inputs), on_unused_input="ignore"
    )

    return lnlike, sample


@pytest.mark.parametrize("marginalize_over_inclination", [True, False])
def test_lnlike_array(marginalize_over_inclination, plot=False):

    # Get the functions
    lnlike, sample = get_functions(
        marginalize_over_inclination=marginalize_over_inclination
    )

    # Generate a dataset
    params = [
        defaults["r"],
        defaults["a"],
        defaults["b"],
        defaults["c"],
        defaults["n"],
        defaults["i"],
        defaults["p"],
    ]
    t = np.linspace(0, 1, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    np.random.seed(42)
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of `b` for definiteness
    ll = np.zeros(100)
    b_arr = np.linspace(0.0, 1.0, len(ll))
    for i in tqdm(range(len(ll))):
        params[2] = b_arr[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    if plot:
        plt.plot(b_arr, np.exp(ll - np.nanmax(ll)))
        plt.show()

    # A *very* simple test
    assert np.abs(b_arr[np.nanargmax(ll)] - defaults["b"]) < 0.10


@pytest.mark.parametrize(
    "param,marginalize_over_inclination",
    product(["r", "a", "b", "c", "n", "i", "p"], [False, True]),
)
def test_lnlike_grad(param, marginalize_over_inclination):

    # Generate a fake dataset
    np.random.seed(42)
    t = np.linspace(0, 3, 100)
    flux = np.random.randn(len(t))
    data_cov = 1.0

    with change_flags(compute_test_value="off"):
        if param in ["i", "p"]:
            verify_grad(
                lambda x: StarryProcess().log_likelihood(
                    t, flux, data_cov, **{param: x}
                ),
                (defaults[param],),
                n_tests=1,
            )
        else:
            verify_grad(
                lambda x: StarryProcess(**{param: x}).log_likelihood(
                    t, flux, data_cov
                ),
                (defaults[param],),
                n_tests=1,
            )

