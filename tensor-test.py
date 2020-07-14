import numpy as np
import starry_process as sp
from scipy.integrate import quad
from scipy.linalg import block_diag, cho_factor
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import gamma
import time


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


def get_basis_integrals(ydeg):
    N = (ydeg + 1) ** 2

    # Integrate the basis terms
    term = np.zeros((4 * ydeg + 1, 4 * ydeg + 1))
    for i in range(4 * ydeg + 1):
        for j in range(0, 4 * ydeg + 1, 2):
            term[i, j] = (
                gamma(0.5 * (i + 1)) * gamma(0.5 * (j + 1)) / gamma(0.5 * (2 + i + j))
            )
    term /= np.pi

    # Moment integrals
    q = np.zeros(N)
    Q = np.zeros((N, N))
    n1 = 0
    for l1 in range(ydeg + 1):
        for m1 in range(-l1, l1 + 1):
            j1 = m1 + l1
            i1 = l1 - m1
            q[n1] = term[i1, j1]
            n2 = 0
            for l2 in range(ydeg + 1):
                for m2 in range(-l2, l2 + 1):
                    j2 = m2 + l2
                    i2 = l2 - m2
                    Q[n1, n2] = term[i1 + i2, j1 + j2]
                    n2 += 1
            n1 += 1

    return q, Q


def get_t(ydeg, R, q):
    t = [np.zeros((2 * ydeg + 1, 2 * ydeg + 1)) for l in range(ydeg + 1)]
    for l in range(ydeg + 1):
        t[l] = R[l] @ q[l ** 2 : (l + 1) ** 2]
    return t


def get_mu(ydeg, t, s):
    N = (ydeg + 1) ** 2
    mu = np.zeros(N)
    for l in range(ydeg + 1):
        i = slice(l ** 2, (l + 1) ** 2)
        mu[i] = t[l] @ s[i]
    return mu


def get_U(ydeg, Q):
    N = (ydeg + 1) ** 2
    n = 2 * ydeg + 1
    w, U = eigh(Q, eigvals=(N - n, N - 1))  # subset_by_index (scipy >= v1.5.0)
    U = U @ np.diag(np.sqrt(np.maximum(0, w)))
    U = U[:, ::-1]
    return U


def get_T(ydeg, R, U):
    T = [np.zeros((2 * ydeg + 1, 2 * ydeg + 1, 2 * ydeg + 1)) for l in range(ydeg + 1)]
    for l in range(ydeg + 1):
        i = slice(l ** 2, (l + 1) ** 2)
        T[l] = np.swapaxes(R[l] @ U[i], 1, 2)
    return T


def get_C(ydeg, T, sqrtS):
    N = (ydeg + 1) ** 2
    sqrtC = np.zeros((N, 2 * ydeg + 1, N))
    for l in range(ydeg + 1):
        i = slice(l ** 2, (l + 1) ** 2)
        sqrtC[i] = T[l] @ sqrtS[i]
    A = sqrtC.reshape(N, -1)
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
R = sp.integrals.wigner.R(ydeg, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0)
q, Q = get_basis_integrals(ydeg)
t = get_t(ydeg, R, q)
U = get_U(ydeg, Q)
T = get_T(ydeg, R, U)
mu_fast = get_mu(ydeg, t, s)
C_fast = get_C(ydeg, T, sqrtS)

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
