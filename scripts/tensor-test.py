import numpy as np
import starry_process as sp
from scipy.integrate import quad
from scipy.linalg import block_diag, cho_factor
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import gamma
import time
import scipy
from packaging import version

# Kewyord to `eigh` changed in 1.5.0
if version.parse(scipy.__version__) < version.parse("1.5.0"):
    eigvals = "eigvals"
else:
    eigvals = "subset_by_index"


def test_rotation_matrix(blockwise=True):
    # Settings
    ydeg = 3
    N = (ydeg + 1) ** 2
    phi = np.pi / 3

    # Vectorized + blockwise
    if blockwise:

        Rijk = sp.integrals.wigner.R(
            ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
        )
        b = np.zeros(N)
        for k in range(N):
            l = np.floor(np.sqrt(k))
            m = k - l ** 2 - l
            j = m + l
            i = l - m
            b[k] = np.sin(phi) ** i * np.cos(phi) ** j
        R1 = block_diag(
            *[Rijk[l].dot(b[l ** 2 : (l + 1) ** 2]) for l in range(ydeg + 1)]
        )

    # Alternatively, we can construct the full matrix and dot it directly
    else:

        R1 = np.zeros((N, N, N))
        b = np.zeros(N)
        for k in range(N):
            l = np.floor(np.sqrt(k))
            m = k - l ** 2 - l
            j = m + l
            i = l - m
            b[k] = np.sin(phi) ** i * np.cos(phi) ** j
        for l in range(ydeg + 1):
            for k in range(2 * l + 1):
                R1[l ** 2 : (l + 1) ** 2, l ** 2 : (l + 1) ** 2, l ** 2 + k] = Rijk[l][
                    :, :, k
                ]
        R1 = R1.dot(b)

    # Direct
    theta = 2 * phi
    R2 = block_diag(
        *sp.integrals.wigner._R(
            ydeg, theta, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
        )
    )

    # Compare
    fig, ax = plt.subplots(1, 3)
    for axis, R in zip(ax, [R1, R2, R1 - R2]):
        vmin = min(np.min(R), -np.max(R))
        vmax = -vmin
        im = axis.imshow(R, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axis)
    plt.show()


class LongitudeIntegral(object):
    def __init__(self, ydeg):
        self.ydeg = ydeg
        self.N = (ydeg + 1) ** 2
        self.n = 2 * ydeg + 1

        # TODO
        self.R = sp.integrals.wigner.R(
            ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0
        )

        # Compute the integrals
        self._compute_basis_integrals()
        self._compute_U()

        # Compute the transforms
        self._compute_t()
        self._compute_T()

    def _get_q_numerical(self):
        def b(phi, n):
            l = np.floor(np.sqrt(n))
            m = n - l ** 2 - l
            j = m + l
            i = l - m
            return np.sin(phi) ** i * np.cos(phi) ** j

        q = np.zeros(self.N)
        for n in range(self.N):
            q[n] = quad(b, 0, 2 * np.pi, args=(n,))[0] / (2 * np.pi)
        return q

    def _get_Q_numerical(self):
        def bbT(phi, n1, n2):
            res = 1.0
            for n in [n1, n2]:
                l = np.floor(np.sqrt(n))
                m = n - l ** 2 - l
                j = m + l
                i = l - m
                res *= np.sin(phi) ** i * np.cos(phi) ** j
            return res

        Q = np.zeros((self.N, self.N))
        for n1 in range(self.N):
            for n2 in range(self.N):
                Q[n1, n2] = quad(bbT, 0, 2 * np.pi, args=(n1, n2))[0] / (2 * np.pi)
        return Q

    def _compute_basis_integrals(self):

        # Integrate the basis terms
        term = np.zeros((4 * self.ydeg + 1, 4 * self.ydeg + 1))
        for i in range(4 * self.ydeg + 1):
            for j in range(0, 4 * self.ydeg + 1, 2):
                term[i, j] = (
                    gamma(0.5 * (i + 1))
                    * gamma(0.5 * (j + 1))
                    / gamma(0.5 * (2 + i + j))
                )
        term /= np.pi

        # Moment integrals
        self.q = np.zeros(self.N)
        self.Q = np.zeros((self.N, self.N))
        n1 = 0
        for l1 in range(self.ydeg + 1):
            for m1 in range(-l1, l1 + 1):
                j1 = m1 + l1
                i1 = l1 - m1
                self.q[n1] = term[i1, j1]
                n2 = 0
                for l2 in range(self.ydeg + 1):
                    for m2 in range(-l2, l2 + 1):
                        j2 = m2 + l2
                        i2 = l2 - m2
                        self.Q[n1, n2] = term[i1 + i2, j1 + j2]
                        n2 += 1
                n1 += 1

    def _compute_U(self):
        w, self.U = eigh(self.Q, **{eigvals: (self.N - self.n, self.N - 1)})
        self.U = self.U @ np.diag(np.sqrt(np.maximum(0, w)))
        self.U = self.U[:, ::-1]

    def _compute_t(self):
        self.t = [np.zeros((self.n, self.n)) for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            self.t[l] = self.R[l] @ self.q[l ** 2 : (l + 1) ** 2]

    def _compute_T(self):
        self.T = [np.zeros((self.n, self.n, self.n)) for l in range(self.ydeg + 1)]
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            self.T[l] = np.swapaxes(self.R[l] @ self.U[i], 1, 2)

    def first_moment(self, s):
        """Compute the first moment of the longitude distribution."""
        mu = np.zeros(self.N)
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            mu[i] = self.t[l] @ s[i]
        return mu

    def second_moment(self, sqrtS):
        """Compute the second moment of the longitude distribution."""
        sqrtC = np.zeros((self.N, self.n, self.N))
        for l in range(self.ydeg + 1):
            i = slice(l ** 2, (l + 1) ** 2)
            sqrtC[i] = self.T[l] @ sqrtS[i]
        A = sqrtC.reshape(self.N, -1)
        C = A @ A.T
        return C


# Settings
np.random.seed(0)
ydeg = 5
N = (ydeg + 1) ** 2
s = np.random.randn(N)
sqrtS = np.tril(np.random.randn(N, N) / N) + np.eye(N) / N
S = sqrtS @ sqrtS.T

# Slow way
L = sp.integrals.longitude.LongitudeIntegral(ydeg)
L.set_vector(s)
L.set_matrix(S)
mu = L.integral1()
C = L.integral2()

# Fast way
L = LongitudeIntegral(ydeg)
mu_fast = L.first_moment(s)
C_fast = L.second_moment(sqrtS)

# Compare
fig, ax = plt.subplots(2, 3, figsize=(16, 7))
ax[0, 0].plot(mu)
ax[0, 1].plot(mu_fast)
ax[0, 2].plot(np.log10(np.abs(mu - mu_fast)))
for axis, X in zip(ax[1], [C, C_fast]):
    vmin = min(np.min(X), -np.max(X))
    vmax = -vmin
    im = axis.imshow(X, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=axis)
cmap = plt.get_cmap("viridis")
cmap.set_under("w")
im = ax[1, 2].imshow(np.log10(np.abs(C - C_fast)), vmin=-15, cmap=cmap)
plt.colorbar(im, ax=ax[1, 2])
plt.show()
