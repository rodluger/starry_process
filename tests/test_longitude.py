from starry_gp.longitude import LongitudeIntegral
from numerical import R
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm


def test_first_moment(ydeg=3):

    # Random vector
    np.random.seed(0)
    N = (ydeg + 1) ** 2
    s = np.random.randn(N)

    # Get analytic integral
    L = LongitudeIntegral(ydeg)
    mu = L.first_moment(s)

    # Integrate numerically
    mu_num = np.zeros(N)
    for n in tqdm(range(N)):

        def func(phi):
            Rl = R(ydeg, phi, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0)
            Rs = np.zeros(N)
            for l in range(ydeg + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                Rs[i] = Rl[l] @ s[i]
            return Rs[n] / (2 * np.pi)

        mu_num[n] = quad(func, 0, 2 * np.pi)[0]

    assert np.allclose(mu, mu_num)


def test_second_moment(ydeg=3):

    # Random matrix
    np.random.seed(0)
    N = (ydeg + 1) ** 2
    sqrtS = np.random.randn(N, N) / N
    S = sqrtS @ sqrtS.T

    # Get analytic integral
    L = LongitudeIntegral(ydeg)
    A = L.second_moment(sqrtS)
    C = A @ A.T

    # Integrate numerically
    C_num = np.zeros((N, N))
    for n1 in tqdm(range(N)):
        for n2 in range(N):

            def func(phi):
                Rl = R(ydeg, phi, cos_alpha=1, sin_alpha=0, cos_gamma=1, sin_gamma=0)
                RSRT = np.zeros((N, N))
                for l1 in range(ydeg + 1):
                    for l2 in range(ydeg + 1):
                        i = slice(l1 ** 2, (l1 + 1) ** 2)
                        j = slice(l2 ** 2, (l2 + 1) ** 2)
                        RSRT[i, j] = Rl[l1] @ S[i, j] @ Rl[l2].T
                return RSRT[n1, n2] / (2 * np.pi)

            C_num[n1, n2] = quad(func, 0, 2 * np.pi)[0]

    assert np.allclose(C, C_num)


if __name__ == "__main__":
    test_first_moment()
    test_second_moment()
