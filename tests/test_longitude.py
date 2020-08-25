from starry_process.longitude import LongitudeIntegral
from starry_process.wigner import R
from starry_process.transforms import get_alpha_beta
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm


def test_longitude(ydeg=3, mu=0.5, nu=0.01):

    # Settings
    np.random.seed(0)
    atol = 1.0e-12

    # Random input moment matrices
    N = (ydeg + 1) ** 2
    s = np.random.randn(N)
    eigS = np.random.randn(N, N) / N
    S = eigS @ eigS.T

    # Get analytic integrals
    I = LongitudeIntegral(ydeg=ydeg)
    e = I._first_moment(s)
    eigE = I._second_moment(eigS)
    E = eigE @ eigE.T

    # Get the first moment by numerical integration
    e_num = np.zeros(N)
    for n in tqdm(range(N)):

        def func(phi):
            Rl = R(
                ydeg,
                phi=phi,
                cos_alpha=1,
                sin_alpha=0,
                cos_gamma=1,
                sin_gamma=0,
            )
            Rs = np.zeros(N)
            for l in range(ydeg + 1):
                i = slice(l ** 2, (l + 1) ** 2)
                Rs[i] = Rl[l] @ s[i]
            return Rs[n] / (2 * np.pi)

        e_num[n] = quad(func, -np.pi, np.pi)[0]

    # Get the second moment by numerical integration
    E_num = np.zeros((N, N))
    for n1 in tqdm(range(N)):
        for n2 in range(N):

            def func(phi):
                Rl = R(
                    ydeg,
                    phi=phi,
                    cos_alpha=1,
                    sin_alpha=0,
                    cos_gamma=1,
                    sin_gamma=0,
                )
                RSRT = np.zeros((N, N))
                for l1 in range(ydeg + 1):
                    for l2 in range(ydeg + 1):
                        i = slice(l1 ** 2, (l1 + 1) ** 2)
                        j = slice(l2 ** 2, (l2 + 1) ** 2)
                        RSRT[i, j] = Rl[l1] @ S[i, j] @ Rl[l2].T

                return RSRT[n1, n2] / (2 * np.pi)

            E_num[n1, n2] = quad(func, 0, 2 * np.pi)[0]

    # Compare
    assert np.allclose(e, e_num, atol=atol), "error in first moment"
    assert np.allclose(E, E_num, atol=atol), "error in second moment"
