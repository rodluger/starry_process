from .integrals import latitude, longitude, size
import numpy as np
import starry
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import beta as Beta
from scipy.stats import poisson
import time


starry.config.lazy = False


class SP(object):
    def __init__(self, ydeg, skip_nullspace=False, **kwargs):
        self.ydeg = ydeg
        self._map = starry.Map(ydeg)
        self._size = size.SizeIntegral(ydeg, skip_nullspace=skip_nullspace)
        self._lat = latitude.LatitudeIntegral(ydeg, skip_nullspace=skip_nullspace)
        self._lon = longitude.LongitudeIntegral(ydeg, skip_nullspace=skip_nullspace)
        self.set_hyperparams(**kwargs)
        self.set_angles(**kwargs)
        self.set_errors(**kwargs)

    def _compute(self):
        if not self._recompute:
            return
        else:
            self._recompute = False

        # Get the GP cov
        self.mu = self._A.dot(self.mu_y)
        self.cov = self._A.dot(self.cov_y.dot(self._A.T))

        # Add the observational uncertainty
        self.cov[np.diag_indices_from(self.cov)] += self._flux_err ** 2

        # Add the prior variance on the flux (multiplicative) baseline
        self.cov += self._flux_mu ** 2

        # Cholesky factorization
        self._cho_cov = cho_factor(self.cov, lower=False)

        # Likelihood normalization
        N = len(self._theta)
        self._norm = -np.sum(np.log(np.diag(self._cho_cov[0]))) - 0.5 * N * np.log(
            2 * np.pi
        )

    def y(self, amp, sigma, lat, lon):
        """Return the Ylm expansion of a spotted star,
           normalized so the mean of the *process* is unity.
        
        Args:
            amp: Amplitude of the spot(s). Positive = dark spot.
            sigma: Spot size(s) (std. dev. of gaussian).
            lat: Spot latitude(s) in degrees.
            lon: Spot longitude(s) in degrees.

        """
        y = np.zeros((self.ydeg + 1) ** 2)
        for k in range(len(amp)):
            self._map[1:, :] = 0.0
            self._map.amp = 1.0
            self._map.add_spot(amp=-amp[k], sigma=sigma[k], lat=lat[k], lon=lon[k])
            y += self._map.amp * self._map.y
        self._map.amp = 1.0
        y[0] = 1 - np.sum(amp)
        return y / y[0]

    def generate_data(
        self,
        N=500,
        alpha=40.0,
        beta=20.0,
        ln_sigma_mu=np.log(0.05),
        ln_sigma_sigma=0.0,
        ln_amp_mu=np.log(0.01),
        ln_amp_sigma=0.0,
        nspot_lambda=10,
        seed=1,
    ):
        """Generate synthetic light curves.
        
        Args:
            N (int, optional): Number of light curves. Defaults to 50.
            alpha (float, optional): Beta distribution parameter. Defaults to 30.0.
            beta (float, optional): Beta distribution parameter. Defaults to 5.0.
            ln_sigma_mu (int, optional): Log spot size mean. Defaults to -3.
            ln_sigma_sigma (float, optional): Log spot size std dev. Defaults to 0.1.
            ln_amp_mu (float, optional): Log spot amplitude mean. Defaults to -2.3.
            ln_amp_sigma (float, optional): Log spot amplitude std dev. Defaults to 0.1.
            nspot_lambda (int, optional): Poisson shape param for number of spots. Defaults to 10.
        
        Returns:
            params, truths, data (dictionaries)
        """

        # Randomizer seed
        np.random.seed(seed)

        # Hyperparams
        params = dict(
            alpha=alpha,
            beta=beta,
            ln_sigma_mu=ln_sigma_mu,
            ln_sigma_sigma=ln_sigma_sigma,
            ln_amp_mu=ln_amp_mu,
            ln_amp_sigma=ln_amp_sigma,
            nspot_lambda=nspot_lambda,
        )

        # Number of spots, their sizes, their amplitudes,
        # their latitudes and longitudes
        nspot = poisson.rvs(params["nspot_lambda"], size=N)
        sigma = np.exp(
            params["ln_sigma_mu"]
            + params["ln_sigma_sigma"] * np.random.randn(nspot.sum())
        )
        x = -np.exp(
            params["ln_amp_mu"] + params["ln_amp_sigma"] * np.random.randn(nspot.sum())
        )
        amp = x / (x - 1)
        lat = (
            np.arccos(Beta.rvs(params["alpha"], params["beta"], size=nspot.sum()))
            * 180.0
            / np.pi
        )
        lat *= 2.0 * (np.array(np.random.random(nspot.sum()) > 0.5, dtype=int) - 0.5)
        lon = 360.0 * np.random.random(nspot.sum())

        # Generate N light curves
        ntheta = len(self._theta)
        flux = np.zeros((ntheta, N))
        y = np.zeros((N, (self.ydeg + 1) ** 2))
        k = slice(0)
        for n in range(N):
            k = slice(k.stop, k.stop + nspot[n])
            y[n] = self.y(amp[k] / nspot[n], sigma[k], lat[k], lon[k])
            flux[:, n] = self._A.dot(y[n]) + self._flux_err * np.random.randn(ntheta)

        # Store everything in dicts
        data = dict(theta=self._theta, flux=flux, flux_err=self._flux_err)
        truths = dict(
            ydeg=self.ydeg, nspot=nspot, sigma=sigma, amp=amp, lat=lat, lon=lon, y=y
        )

        return params, truths, data

    def set_hyperparams(
        self,
        alpha=40.0,
        beta=20.0,
        ln_sigma_mu=np.log(0.05),
        ln_sigma_sigma=0.0,
        ln_amp_mu=np.log(0.01),
        ln_amp_sigma=0.0,
        sign=-1,
        **kwargs
    ):
        self._size.set_params(
            ln_sigma_mu, ln_sigma_sigma, ln_amp_mu, ln_amp_sigma, sign
        )
        self._lat.set_params(alpha, beta)

        # Compute the y mean
        vector = self._size.integral1()
        vector = self._lat.integral1(vector=vector)
        self.mu_y = self._lon.integral1(vector=vector)

        # Compute the y covariance
        matrix = self._size.integral2()
        matrix = self._lat.integral2(matrix=matrix)
        EyyT = self._lon.integral2(matrix=matrix)
        self.cov_y = EyyT - self.mu_y.reshape(-1, 1).dot(self.mu_y.reshape(1, -1))
        self._cho_cov_y = cho_factor(self.cov_y, lower=True)
        self._norm_y = -np.sum(np.log(np.diag(self._cho_cov_y[0]))) - 0.5 * len(
            self.mu_y
        ) * np.log(2 * np.pi)

        self._recompute = True

    def set_angles(
        self, theta=np.linspace(0, 360, 100, endpoint=False), inc=60, **kwargs
    ):
        self._map.inc = inc
        self._theta = theta
        self._A = self._map.design_matrix(theta=theta)
        self._recompute = True

    def set_errors(self, flux_err=1e-5, flux_mu=1e-2, **kwargs):
        self._flux_err = flux_err
        self._flux_mu = flux_mu
        self._recompute = True

    def log_likelihood(self, flux=None, y=None):
        assert (flux is not None and y is None) or (
            flux is None and y is not None
        ), "Please provide *either* `flux` or `y`."
        if flux is not None:
            self._compute()
            CInvflux = cho_solve(self._cho_cov, flux)
            return np.sum(
                [
                    -0.5 * f.dot(CInvf) + self._norm
                    for f, CInvf in zip(flux.T, CInvflux.T)
                ]
            )
        else:
            r = y - self.mu_y
            CInvy = cho_solve(self._cho_cov_y, y)
            return r.T.dot(CInvy) + self._norm_y

    def draw_y(self, ndraws=1):
        npts = self.cov_y.shape[0]
        L = np.tril(self._cho_cov_y[0])
        u = np.random.randn(npts, ndraws)
        x = np.dot(L, u) + self.mu_y[:, None]
        return x.T

    def _test_mu_y_and_cov_y(self):
        # NOTE: disable `skip_nullspace` for this test

        alpha = 40.0
        beta = 20.0
        ln_sigma_mu = np.log(0.1)
        ln_sigma_sigma = 0.0
        ln_amp_mu = np.log(0.2)
        ln_amp_sigma = 0.0
        self.set_hyperparams(
            alpha, beta, ln_sigma_mu, ln_sigma_sigma, ln_amp_mu, ln_amp_sigma
        )

        x = -np.exp(ln_amp_mu)
        amp = x / (x - 1)
        sigma = np.exp(ln_sigma_mu)
        nsamples = 100000
        np.random.seed(1)
        y = np.zeros((nsamples, (self.ydeg + 1) ** 2))
        for n in range(nsamples):
            lat = np.arccos(Beta.rvs(alpha, beta)) * 180.0 / np.pi
            lat *= 2.0 * (int(np.random.random() > 0.5) - 0.5)
            lon = 360.0 * np.random.random()
            y[n] = self.y([amp], [sigma], [lat], [lon])
        mu_y_num = np.mean(y, axis=0)
        cov_y_num = np.cov(y.T)

        # Compare
        assert np.allclose(mu_y_num, self.mu_y, atol=1e-3)
        assert np.allclose(cov_y_num, self.cov_y, atol=1e-3)

    def _test_nullspace(self):

        alpha = 2.0
        beta = 2.0
        sigma_mu = 0.01
        sigma_sigma = 0.0
        amp_mu = -0.1
        amp_sigma = 0.0
        theta = np.linspace(0, 360, 50)
        inc = 60
        self.set_angles(theta, inc=inc)

        # Compute everything
        self._size.ls = np.arange(self.ydeg + 1)
        self._lat.ls = np.arange(self.ydeg + 1)
        self._lon.ls = np.arange(self.ydeg + 1)
        self.set_hyperparams(alpha, beta, sigma_mu, sigma_sigma, amp_mu, amp_sigma)
        self._compute()
        mu1 = np.array(self.mu)
        cov1 = np.array(self.cov)

        # Skip the nullspace
        self._size.ls = np.append(self._size.ls[:3], self._size.ls[4::2])
        self._lat.ls = np.append(self._lat.ls[:3], self._lat.ls[4::2])
        self._lon.ls = np.append(self._lon.ls[:3], self._lon.ls[4::2])
        self.set_hyperparams(alpha, beta, sigma_mu, sigma_sigma, amp_mu, amp_sigma)
        self._compute()
        mu2 = np.array(self.mu)
        cov2 = np.array(self.cov)

        assert np.allclose(mu1, mu2)
        assert np.allclose(cov1, cov2)

    def _test_time(self):

        alpha = 2.0
        beta = 2.0
        sigma_mu = -3
        sigma_sigma = 0.0
        amp_mu = -2.3
        amp_sigma = 0.0
        theta = np.linspace(0, 360, 1000)
        inc = 60
        self.set_angles(theta, inc=inc)

        # Time it
        ntimes = 10
        tstart = time.time()
        for i in range(ntimes):
            self.set_hyperparams(alpha, beta, sigma_mu, sigma_sigma, amp_mu, amp_sigma)
            self._compute()
        tend = time.time()
        ttotal = (tend - tstart) / ntimes

        print("%.5f s per iteration" % ttotal)
