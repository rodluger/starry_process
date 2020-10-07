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


def get_functions():
    def _lnlike(
        sa, sb, la, lb, ca, cb, period, inc, t, flux, data_cov,
    ):
        gp = StarryProcess(
            size=[sa, sb], latitude=[la, lb], contrast=[ca, cb],
        )
        return gp.log_likelihood(t, flux, data_cov, period=period, inc=inc)

    def _sample(
        sa, sb, la, lb, ca, cb, period, inc, t,
    ):
        gp = StarryProcess(size=[sa, sb], latitude=[la, lb], contrast=[ca, cb])
        gp.random.seed(42)
        return tt.reshape(gp.sample(t, period=period, inc=inc), (-1,))

    # Likelihood func
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector(), tt.dvector(), tt.dscalar()]
    lnlike = theano.function(inputs, _lnlike(*inputs))

    # Draw a sample
    inputs = [tt.dscalar() for n in range(8)]
    inputs += [tt.dvector()]
    sample = theano.function(inputs, _sample(*inputs))

    return lnlike, sample


def test_lnlike_array(plot=False):

    # Get the functions
    lnlike, sample = get_functions()

    # Generate a dataset
    params = [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        10.0,
        1.0,
        60.0,
    ]
    t = np.linspace(0, 3, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    np.random.seed(42)
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of `lb` for definiteness
    ll = np.zeros(100)
    lb_arr = np.linspace(0.0, 1.0, len(ll))
    for i in range(len(ll)):
        params[3] = lb_arr[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    if plot:
        plt.plot(lb_arr, np.exp(ll - np.nanmax(ll)))
        plt.show()

    # A *very* simple test
    assert np.abs(lb_arr[np.nanargmax(ll)] - 0.5) < 0.10


@pytest.mark.parametrize(
    "param", ["l", "la", "lb", "s", "sa", "sb", "ca", "cb", "period", "inc"]
)
def test_lnlike_grad(param):

    # Generate a fake dataset
    np.random.seed(42)
    t = np.linspace(0, 3, 100)
    flux = np.random.randn(len(t))
    data_cov = 1.0

    with change_flags(compute_test_value="off"):
        if param in ["period", "inc"]:
            if param == "period":
                value = 1.0
            else:
                value = 60.0
            verify_grad(
                lambda x: StarryProcess().log_likelihood(
                    t, flux, data_cov, **{param: x}
                ),
                (value,),
                n_tests=1,
            )
        else:
            if param in ["s", "sa", "sb"]:
                key = "size"
                if param == "s":
                    f = lambda x: x
                    value = 20.0
                elif param == "sa":
                    f = lambda x: [x, 0.5]
                    value = 0.5
                else:
                    f = lambda x: [0.5, x]
                    value = 0.5
            elif param in ["l", "la", "lb"]:
                key = "latitude"
                if param == "l":
                    f = lambda x: x
                    value = 20.0
                elif param == "la":
                    f = lambda x: [x, 0.5]
                    value = 0.5
                else:
                    f = lambda x: [0.5, x]
                    value = 0.5
            else:
                key = "contrast"
                if param == "ca":
                    f = lambda x: [x, 10.0]
                    value = 0.5
                else:
                    f = lambda x: [0.5, x]
                    value = 10.0
            verify_grad(
                lambda x: StarryProcess(**{key: f(x)}).log_likelihood(
                    t, flux, data_cov
                ),
                (value,),
                n_tests=1,
            )

