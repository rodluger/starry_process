import numpy as np
from scipy.special import hyp2f1
from starry_gp.transform import get_c0_c1


def get_G_numerical(ydeg=10, alpha=5.0, beta=3.3):
    """
    Compute the G_j^k matrix numerically.
    
    """
    c0, c1 = get_c0_c1(ydeg)
    z = -c1 / (1 + c0)
    G = np.empty((2 * ydeg + 2, 5))
    for j in range(2 * ydeg + 2):
        for k in range(5):
            G[j, k] = hyp2f1(j + 1, alpha + k, alpha + beta + k, z)
    return G


def get_G(ydeg=10, alpha=5.0, beta=3.3):
    """
    Compute the G_j^k matrix recursively.

    """
    # Parameters
    c0, c1 = get_c0_c1(ydeg)
    z = -c1 / (1 + c0)
    ab = alpha + beta
    lam = np.array(
        [
            alpha / ab,
            (alpha + 1) / (ab + 1),
            (alpha + 2) / (ab + 2),
            (alpha + 3) / (ab + 3),
        ]
    )

    # Hypergeometric sequence
    G = np.empty((2 * ydeg + 2, 5))

    # Compute the first four terms explicitly
    G[0, 0] = hyp2f1(1, alpha, ab, z)
    G[0, 1] = hyp2f1(1, alpha + 1, ab + 1, z)
    G[1, 0] = hyp2f1(2, alpha, ab, z)
    G[1, 1] = hyp2f1(2, alpha + 1, ab + 1, z)

    # Recurse upward in k
    for j in range(2):
        for k in range(2, 5):
            A = (ab + k - 2) * (1 + c0)
            B = (
                alpha
                - (alpha + k) * c1
                + beta
                + k
                + (ab + k - 2) * c0
                + (j + 2) * c1
                - 2
            )
            C = lam[k - 1] * (ab - j + k - 2) * c1
            G[j, k] = (A * G[j, k - 2] - B * G[j, k - 1]) / C

    # Now recurse upward in j
    for j in range(2, 2 * ydeg + 2):
        for k in range(5):
            A = (ab + k - j) * (1 + c0)
            B = (ab - 2 * j + k) * (1 + c0) + (alpha - j + k) * c1
            C = j * (1 + c0 + c1)
            G[j, k] = (A * G[j - 2, k] - B * G[j - 1, k]) / C

    return G


def test_hypgeo(ydeg=10, alpha=5.0, beta=3.3):
    """
    Verify that our recursion for G_j^k is correct.

    """
    G = get_G(ydeg=ydeg, alpha=alpha, beta=beta)
    Gnum = get_G_numerical(ydeg=ydeg, alpha=alpha, beta=beta)
    assert np.allclose(G, Gnum, atol=1e-15, rtol=1e-15)
