from starry_process.longitude import LongitudeIntegral
from starry_process.wigner import R
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm


def test_longitude(ydeg=3, rtol=1e-12, ftol=1e-10):

    # Random input moment matrices
    np.random.seed(0)
    N = (ydeg + 1) ** 2
    s = np.random.randn(N)
    eigS = np.random.randn(N, N) / N
    S = eigS @ eigS.T

    # Get analytic integrals
    print("Computing moments analytically...")
    I = LongitudeIntegral(ydeg=ydeg)
    e = I._first_moment(s).eval()
    eigE = I._second_moment(eigS).eval()
    E = eigE @ eigE.T

    # Get the first moment by numerical integration
    e_num = np.zeros(N)
    print("Computing first moment numerically...")
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
    print("Computing second moment numerically...")
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

    # Avoid div by zero in the comparison
    nonzero_i = np.abs(e_num) > 1e-15
    nonzero_ij = np.abs(E_num) > 1e-15

    # Compare
    assert np.max(np.abs(e - e_num)) < rtol, "error in first moment"
    assert (
        np.max(np.abs(1 - e[nonzero_i] / e_num[nonzero_i])) < ftol
    ), "error in first moment"
    assert np.max(np.abs(E - E_num)) < rtol, "error in second moment"
    assert (
        np.max(np.abs(1 - E[nonzero_ij] / E_num[nonzero_ij])) < ftol
    ), "error in second moment"
