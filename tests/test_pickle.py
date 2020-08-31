from starry_process import StarryProcess
from six.moves import cPickle
import theano
import theano.tensor as tt
import sys
import os
import pytest


@pytest.mark.dependency()
def test_save():

    # Parameters
    ydeg = 15
    size_alpha = tt.dscalar()
    size_beta = tt.dscalar()
    latitude_alpha = tt.dscalar()
    latitude_beta = tt.dscalar()
    contrast_mu = tt.dscalar()
    contrast_sigma = tt.dscalar()
    period = tt.dscalar()
    inc = tt.dscalar()
    t = tt.dvector()

    # Set up the process
    gp = StarryProcess(ydeg)
    gp.size.set_params(size_alpha, size_beta)
    gp.latitude.set_params(latitude_alpha, latitude_beta)
    gp.contrast.set_params(contrast_mu, contrast_sigma)
    gp.design.set_params(period, inc)

    # Compile a function that computes the mean and covariance
    function = theano.function(
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
        [gp.mean(t), gp.cov(t)],
    )

    # Pickle it!
    sys.setrecursionlimit(2000)
    with open("sp.save", "wb") as f:
        cPickle.dump(function, f, protocol=cPickle.HIGHEST_PROTOCOL)


@pytest.mark.dependency(depends=["test_save"])
def test_load():

    # Load it
    with open("sp.save", "rb") as f:
        function = cPickle.load(f)

    # Run the function and just check the shapes it returns
    mean, cov = function(1, 50, 10, 30, 0.5, 0.1, 3, 65, [0, 1, 2, 3])
    assert mean.shape == (4,)
    assert cov.shape == (4, 4)

    # Delete the pickle
    os.remove("sp.save")
