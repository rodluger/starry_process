from starry_process.gp import YlmGP
import theano
import theano.tensor as tt
import timeit


def test_timing(ydeg=15):

    # Free parameters
    size_alpha = tt.dscalar()
    size_beta = tt.dscalar()
    contrast_mu = tt.dscalar()
    contrast_sigma = tt.dscalar()
    latitude_alpha = tt.dscalar()
    latitude_beta = tt.dscalar()

    # Compute the mean and covariance
    gp = YlmGP(ydeg)
    gp.size.set_params(size_alpha, size_beta)
    gp.contrast.set_params(contrast_mu, contrast_sigma)
    gp.latitude.set_params(latitude_alpha, latitude_beta)
    mu = gp.mean
    cov = gp.cov

    # Compile the function
    get_mu_and_cov = theano.function(
        [
            size_alpha,
            size_beta,
            contrast_mu,
            contrast_sigma,
            latitude_alpha,
            latitude_beta,
        ],
        [mu, cov],
    )

    # Time it!
    number = 100
    time = (
        timeit.timeit(
            lambda: get_mu_and_cov(10.0, 30.0, 0.5, 0.1, 10.0, 50.0),
            number=number,
        )
        / number
    )

    print("time elapsed: {:.4f} s".format(time))
    assert time < 0.1, "too slow!"
