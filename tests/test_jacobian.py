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
            sp = StarryProcess(latitude=[a, b])
            transform = sp.latitude._transform.inverse_transform
            pdf = sp.latitude.pdf
            m1, m2 = 15, 75
            s1, s2 = 10, 30
            xmin = -89
            xmax = 89
        else:
            sp = StarryProcess(size=[a, b])
            transform = sp.size._transform.inverse_transform
            pdf = sp.size.pdf
            m1, m2 = 30, 60
            s1, s2 = 7, 20
            xmin = 0
            xmax = 75
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
            plt.figure()
            ndraws = 1000
            idx = np.random.choice(len(samples), size=(ndraws,))
            x = np.linspace(xmin, xmax, 1000)
            p = np.empty((ndraws, len(x)))
            for i in tqdm(range(ndraws)):
                p[i] = pdf(x, a=a[idx[i]], b=b[idx[i]])
                plt.plot(x, p[i], color="C0", lw=1, alpha=0.1)
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


if __name__ == "__main__":
    test_latitude_jacobian(plot=True)
