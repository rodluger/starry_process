from starry_process import StarryProcess
from starry_process.latitude import beta2gauss
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymc3 as pm
import exoplanet as xo
import theano
import theano.tensor as tt
from tqdm import tqdm
from corner import corner
from scipy.stats import gaussian_kde
from scipy.stats import median_abs_deviation as mad


def test_jacobian(plot=False):

    # Compile the PDF
    _x = tt.dvector()
    _a = tt.dscalar()
    _b = tt.dscalar()
    pdf = theano.function(
        [_x, _a, _b], StarryProcess(a=_a, b=_b).latitude.pdf(_x)
    )

    with pm.Model() as model:

        # Uniform sampling in `a` and `b`
        a = pm.Uniform("a", 0, 1)
        b = pm.Uniform("b", 0, 1)

        # Likelihood w/ no data: just the prior!
        sp = StarryProcess(a=a, b=b)
        m1, m2 = 0, 80
        s1, s2 = 0, 45
        xmin = -90
        xmax = 90
        pm.Potential("jacobian", sp.log_jac())

        # Sample
        # NOTE: Sampling straight from this prior is really tough because
        # it has really high curvature in some places. Typically
        # about half of the samples end in divergences! (This is much less of
        # an issue when we have data.) Despite these issues, the test still
        # works: the posterior density in `mu` and `sigma` is quite uniform.
        trace = pm.sample(
            tune=1000,
            draws=25000,
            chains=4,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )

        # Transform samples to `mu`, `sigma`
        samples = np.array(pm.trace_to_dataframe(trace))
        a, b = samples.T
        mu, sigma = beta2gauss(a, b)
        tr_samples = np.transpose([mu, sigma])

        if plot:
            corner(tr_samples, plot_density=False, plot_contours=False)
            plt.figure()
            ndraws = 1000
            idx = np.random.choice(len(samples), size=(ndraws,))
            x = np.linspace(xmin, xmax, 1000)
            p = np.empty((ndraws, len(x)))
            for i in tqdm(range(ndraws)):
                p[i] = pdf(x, a[idx[i]], b[idx[i]])
                plt.plot(x, p[i], color="C0", lw=1, alpha=0.1)
            plt.show()

        # Approximate the density with a Gaussian KDE
        # and check that the variation is < 10%
        kernel = gaussian_kde(tr_samples.T)
        X, Y = np.mgrid[m1:m2:100j, s1:s2:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = np.reshape(kernel(positions).T, X.shape).T
        std = 1.4826 * mad(density.flatten())
        mean = np.mean(density.flatten())
        assert std / mean < 0.1


if __name__ == "__main__":
    test_jacobian(plot=True)
