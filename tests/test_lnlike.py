from starry_process import StarryProcess
from starry_process.defaults import defaults
import numpy as np
import theano.tensor as tt
import theano
import pymc3 as pm
import exoplanet
import matplotlib.pyplot as plt


def get_functions():
    def _lnlike(
        sa, sb, la, lb, ca, cb, period, inc, t, flux, data_cov,
    ):
        gp = StarryProcess(
            sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, period=period, inc=inc,
        )
        return gp.log_likelihood(t, flux, data_cov)

    def _sample(
        sa, sb, la, lb, ca, cb, period, inc, t,
    ):
        gp = StarryProcess(
            sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb, period=period, inc=inc,
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


def test_lnlike_array(plot=False):

    # Get the functions
    lnlike, sample = get_functions()

    # Generate a dataset
    params = [
        defaults["sa"],
        defaults["sb"],
        defaults["la"],
        defaults["lb"],
        defaults["ca"],
        defaults["cb"],
        defaults["period"],
        defaults["inc"],
    ]
    t = np.linspace(0, 3, 1000)
    flux = sample(*params, t)
    flux_err = 1e-6
    data_cov = flux_err ** 2
    np.random.seed(42)
    flux += np.random.randn(len(t)) * flux_err

    # Compute the pdf of `lb`
    ll = np.zeros(100)
    lb_arr = np.linspace(0.0, 1.0, len(ll))
    for i in range(len(ll)):
        params[3] = lb_arr[i]
        ll[i] = lnlike(*params, t, flux, data_cov)

    if plot:
        plt.plot(lb_arr, np.exp(ll - np.nanmax(ll)))
        plt.show()

    # A *very* simple test
    assert np.abs(lb_arr[np.nanargmax(ll)] - defaults["lb"]) < 0.10


if __name__ == "__main__":
    test_lnlike_array()
