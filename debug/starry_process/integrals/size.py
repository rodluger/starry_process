import numpy as np


class SizeIntegral(object):
    # NOTE: The prior on the amplitude is actually
    # a prior on the quantity `amp / (amp - 1)`, which
    # is the *normalized* amplitude.
    def __init__(self, ydeg, skip_nullspace=False):
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        self.ls = np.arange(self.ydeg + 1)
        if skip_nullspace:
            self.ls = np.append(self.ls[:3], self.ls[4::2])
        self.set_params(-3, 0, -2.3, 0, -1)
        self._precompute()

    @property
    def sigma_mu(self):
        return self._sigma_mu

    @property
    def sigma_sigma(self):
        return self._sigma_sigma

    @property
    def amp_mu(self):
        return self._amp_mu

    @property
    def amp_sigma(self):
        return self._amp_sigma

    def _precompute(self):
        # Coefficients of the polynomial expansion of sigma for each Ylm
        self._term = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # The Ylm coefficient at index n(n + 1) is
        #   amp * self._term[n].dot(basis)
        # where basis is [1, sigma, sigma ** 2, ...]

        # Compute the integrals recursively
        IP = np.zeros((self.ydeg + 1, self.ydeg + 1))
        ID = np.zeros((self.ydeg + 1, self.ydeg + 1))

        # Seeding values
        IP[0, 0] = 1.0
        IP[1, 0] = 1.0
        IP[1, 1] = -np.sqrt(2 / np.pi)
        ID[1, 0] = 1.0
        self._term[0] = IP[0]
        self._term[1] = np.sqrt(3) * IP[1]

        # Recurse
        for n in range(2, self.ydeg + 1):
            C = 2.0 * n - 1.0
            A = C / n
            B = A - 1
            IP[n] = A * np.roll(ID[n - 1], 2) + A * IP[n - 1] - B * IP[n - 2]
            IP[n, 1] -= A * np.sqrt(2 / np.pi)
            ID[n] = C * IP[n - 1] + ID[n - 2]
            self._term[n] = np.sqrt(2 * n + 1) * IP[n]

    def set_params(self, sigma_mu, sigma_sigma, amp_mu, amp_sigma, sign):
        self._sigma_mu = sigma_mu
        self._sigma_sigma = sigma_sigma
        self._amp_mu = amp_mu
        self._amp_sigma = amp_sigma
        self._sign = sign

        # Sigma Taylor basis
        n = np.arange(2 * self.ydeg + 1)
        tmp = np.exp(n * self.sigma_mu + 0.5 * (n * self.sigma_sigma) ** 2)
        self._sigma_basis_vec = tmp[: self.ydeg + 1]
        self._sigma_basis_mat = np.zeros((self.ydeg + 1, self.ydeg + 1))
        for n in range(self.ydeg + 1):
            for m in range(self.ydeg + 1):
                self._sigma_basis_mat[n, m] = tmp[n + m]

        # Amplitude integrals
        self._amp_factor1 = self._sign * np.exp(self.amp_mu + 0.5 * self.amp_sigma ** 2)
        self._amp_factor2 = np.exp(2 * self.amp_mu + 2 * self.amp_sigma ** 2)

    def _s(self, sigma, amp, sign):
        # Compute the integrals recursively
        IP = np.zeros(self.ydeg + 1)
        ID = np.zeros(self.ydeg + 1)
        s = np.zeros((self.ydeg + 1) * (self.ydeg + 1))

        # Seeding values
        IP[0] = 1.0
        IP[1] = 1.0 - sigma * np.sqrt(2 / np.pi)
        ID[0] = 0.0
        ID[1] = 1.0
        s[0] = 1.0 - amp
        s[2] = sign * amp * np.sqrt(3) * IP[1]

        # Recurse
        for n in range(2, self.ydeg + 1):
            c = 2.0 * n - 1.0
            a = c / n
            b = a - 1
            IP[n] = (
                a * sigma ** 2 * ID[n - 1]
                - a * sigma * np.sqrt(2 / np.pi)
                + a * IP[n - 1]
                - b * IP[n - 2]
            )
            ID[n] = c * IP[n - 1] + ID[n - 2]
            s[n * n + n] = sign * amp * np.sqrt(2 * n + 1) * IP[n]

        return s / s[0]

    def integral1(
        self, sigma_mu=None, sigma_sigma=None, amp_mu=None, amp_sigma=None, sign=None
    ):
        if (
            sigma_mu is not None
            or sigma_sigma is not None
            or amp_mu is not None
            or amp_sigma is not None
            or sign is not None
        ):
            self.set_params(sigma_mu, sigma_sigma, amp_mu, amp_sigma, sign)
        S = np.zeros(self.N)
        inds = self.ls * (self.ls + 1)
        S[inds] = self._amp_factor1 * self._term.dot(self._sigma_basis_vec)
        S[0] = 1.0
        return S

    def _test_vectorization(self):
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        S = self.integral1()
        self.ls = ls
        sigma = np.exp(self.sigma_mu)
        x = -np.exp(self.amp_mu)
        amp = x / (x - 1)
        assert np.allclose(S, self._s(sigma, amp, self._sign))

    def _test_integral1(self, sigma_sigma=0.1, amp_sigma=0.5):
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        S = self.integral1(
            sigma_mu=-3, sigma_sigma=sigma_sigma, amp_mu=-2.3, amp_sigma=amp_sigma
        )
        self.ls = ls
        np.random.seed(1)
        nsamples = 100000
        lnsigma = self.sigma_mu + self.sigma_sigma * np.random.randn(nsamples)
        sigma = np.exp(lnsigma)
        lnamp = self.amp_mu + self.amp_sigma * np.random.randn(nsamples)
        x = -np.exp(lnamp)
        amp = x / (x - 1)
        s = np.zeros(self.N)
        for k in range(nsamples):
            s += self._s(sigma[k], amp[k]) / nsamples
        assert np.allclose(s, S, atol=1e-3)

    def integral2(self, sigma_mu=None, sigma_sigma=None, amp_mu=None, amp_sigma=None, sign=None):
        if (
            sigma_mu is not None
            or sigma_sigma is not None
            or amp_mu is not None
            or amp_sigma is not None
            or sign is not None
        ):
            self.set_params(sigma_mu, sigma_sigma, amp_mu, amp_sigma, sign)
        S = np.zeros((self.N, self.N))
        for l1 in self.ls:
            i = l1 * (l1 + 1)
            u = self._term[l1].reshape(1, -1)
            uS = u.dot(self._sigma_basis_mat)
            for l2 in self.ls:
                j = l2 * (l2 + 1)
                v = self._term[l2].reshape(-1, 1)
                S[i, j] = uS.dot(v)
        S *= self._amp_factor2
        S[0, :] = self.integral1()
        S[:, 0] = S[0, :]
        return S

    def _test_integral2(self, sigma_sigma=0.1, amp_sigma=0.5, sign=1):
        ls = np.array(self.ls)
        self.ls = np.arange(self.ydeg + 1)
        S = self.integral2(
            sigma_mu=-3, sigma_sigma=sigma_sigma, amp_mu=-2.3, amp_sigma=amp_sigma, sign=sign
        )
        self.ls = ls
        np.random.seed(1)
        nsamples = 100000
        lnsigma = self.sigma_mu + self.sigma_sigma * np.random.randn(nsamples)
        sigma = np.exp(lnsigma)
        lnamp = self.amp_mu + self.amp_sigma * np.random.randn(nsamples)
        x = -np.exp(lnamp)
        amp = x / (x - 1)
        s = np.zeros((nsamples, self.N))
        for k in range(nsamples):
            s[k] = self._s(sigma[k], amp[k], sign)
        mu = np.mean(s, axis=0).reshape(-1, 1)
        Snum = np.cov(s.T) + mu.dot(mu.T)
        assert np.allclose(S, Snum, atol=1e-3)


def test():
    integral = SizeIntegral(4)
    integral._test_vectorization()
    integral._test_integral1()
    integral._test_integral2()
