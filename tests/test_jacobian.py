from starry_process import StarryProcess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymc3 as pm
import exoplanet as xo
from tqdm import tqdm
from corner import corner
from scipy.stats import gaussian_kde


def sample(param="latitude", plot=False):

    with pm.Model() as model:

        # Uniform sampling in `a` and `b`
        a = pm.Uniform("a", 0, 1)
        b = pm.Uniform("b", 0, 1)

        # Likelihood w/ no data: just the prior!
        if param == "latitude":
            sp = StarryProcess(la=a, lb=b)
            transform = sp.latitude.transform.inverse_transform
            m1, m2 = 15, 75
            s1, s2 = 10, 30
        else:
            sp = StarryProcess(sa=a, sb=b)
            transform = sp.size.transform.inverse_transform
            m1, m2 = 30, 60
            s1, s2 = 7, 20
        pm.Potential("jacobian", sp.log_jac())

        # Sample
        trace = pm.sample(
            tune=500,
            draws=5000,
            chains=4,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )

        # Transform samples to `mu`, `sigma`
        samples = np.array(pm.trace_to_dataframe(trace))
        a, b = samples.T
        tr_samples = np.zeros_like(samples)
        for k in tqdm(range(len(samples))):
            tr_samples[k] = transform(a[k], b[k])

        if plot:
            corner(tr_samples, plot_density=False, plot_contours=False)
            plt.show()

        # Approximate the density with a Gaussian KDE
        # and check that the variation is < 10%
        kernel = gaussian_kde(tr_samples.T)
        X, Y = np.mgrid[m1:m2:100j, s1:s2:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = np.reshape(kernel(positions).T, X.shape).T
        assert np.std(density) / np.mean(density) < 0.1


def test_latitude_jacobian(**kwargs):
    sample("latitude", **kwargs)


def test_size_jacobian(**kwargs):
    sample("size", **kwargs)
