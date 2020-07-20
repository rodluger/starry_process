import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from starry_process.integrals import wigner
import starry
import time


def prod(x1, x2):
    l1 = (len(x1) - 1) // 2
    l2 = (len(x2) - 1) // 2
    l = l1 + l2
    result = np.zeros(2 * l + 1)
    for m, x1_m in enumerate(x1):
        for n, x2_n in enumerate(x2):
            result[m + n] += x1_m * x2_n
    return result


def shift_left(x):
    return np.append(x[1:], [0])


def dlmn(l, s1, c1, s3, c3, D, R):

    iinf = 1 - l
    isup = -iinf

    # Compute the D[lm',m) matrix.
    # First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    D[l][4 * l * (l + 1)] = prod(D[l - 1][2 * l * (isup + l - 1)], [0, 0, 1])
    D[l][2 * l * (2 * l + 1)] = prod(
        D[l - 1][(2 * l - 1) * (isup + l - 1) - isup + l - 1], [1, 0, 0]
    )
    for m in range(isup, iinf - 1, -1):
        # Multiplication by s/c
        D[l][2 * l * (2 * l + 1) + m + l] = shift_left(
            -np.sqrt((l + m + 1.0) / (l - m)) * D[l][2 * l * (2 * l + 1) + l + m + 1]
        )

    # The rows of the upper quarter triangle of the D[lm',m) matrix
    # (Eq. 21 in Alvarez Collado et al.)
    for mp in range(l - 1, -1, -1):
        laux = l + mp
        lbux = l - mp
        aux = 1.0 / ((l - 1) * np.sqrt(laux * lbux))
        cux = np.sqrt((laux - 1) * (lbux - 1)) * l
        for m in range(isup, iinf - 1, -1):
            lauz = l + m
            lbuz = l - m
            auz = 1.0 / np.sqrt(lauz * lbuz)
            fact = aux * auz
            a = l * (l - 1)
            b = -(m * mp) / a
            D[l][(2 * l + 1) * (mp + l) + m + l] = prod(
                fact
                * (2 * l - 1)
                * a
                * D[l - 1][(2 * l - 1) * (mp + l - 1) + m + l - 1],
                [b - 1, 0, b + 1],
            )
            if (lbuz != 1) and (lbux != 1):
                # Trick: Promote D[l - 2] to D[l] by multiplying by (c^2 + s^2)^2
                cuz = np.sqrt(((lauz - 1) * (lbuz - 1)))
                D[l][(2 * l + 1) * (mp + l) + m + l] -= (fact * cux * cuz) * prod(
                    D[l - 2][(2 * l - 3) * (mp + l - 2) + m + l - 2], [1, 0, 2, 0, 1],
                )
        iinf += 1
        isup -= 1

    # The remaining elements of the D[lm',m) matrix are calculated
    # using the corresponding symmetry relations:
    # reflection ---> ((-1)**(m-m')) D[lm,m') = D[lm',m), m'<=m
    # inversion ---> ((-1)**(m-m')) D[l-m',-m) = D[lm',m)

    # Reflection
    sign = 1
    iinf = -l
    isup = l - 1
    for m in range(l, 0, -1):
        for mp in range(iinf, isup + 1):
            D[l][(2 * l + 1) * (mp + l) + m + l] = (
                sign * D[l][(2 * l + 1) * (m + l) + mp + l]
            )
            sign *= -1
        iinf += 1
        isup -= 1

    # Inversion
    iinf = -l
    isup = iinf
    for m in range(l - 1, -(l + 1), -1):
        sign = -1
        for mp in range(isup, iinf - 1, -1):
            D[l][(2 * l + 1) * (mp + l) + m + l] = (
                sign * D[l][(2 * l + 1) * (-mp + l) - m + l]
            )
            sign *= -1
        isup += 1

    # Compute the real rotation matrices R from the complex ones D
    R[l][(2 * l + 1) * l + l] = D[l][(2 * l + 1) * l + l]
    cosmal = c1
    sinmal = s1
    sign = -1
    root_two = np.sqrt((2.0))
    for mp in range(1, l + 1):
        cosmga = c3
        sinmga = s3
        aux = root_two * D[l][(2 * l + 1) * l + mp + l]
        R[l][(2 * l + 1) * (mp + l) + l] = aux * cosmal
        R[l][(2 * l + 1) * (-mp + l) + l] = aux * sinmal
        for m in range(1, l + 1):
            aux = root_two * D[l][(2 * l + 1) * (m + l) + l]
            R[l][(2 * l + 1) * l + m + l] = aux * cosmga
            R[l][(2 * l + 1) * l - m + l] = -aux * sinmga
            d1 = D[l][(2 * l + 1) * (-mp + l) - m + l]
            d2 = sign * D[l][(2 * l + 1) * (mp + l) - m + l]
            cosag = cosmal * cosmga - sinmal * sinmga
            cosagm = cosmal * cosmga + sinmal * sinmga
            sinag = sinmal * cosmga + cosmal * sinmga
            sinagm = sinmal * cosmga - cosmal * sinmga
            R[l][(2 * l + 1) * (mp + l) + m + l] = d1 * cosag + d2 * cosagm
            R[l][(2 * l + 1) * (mp + l) - m + l] = -d1 * sinag + d2 * sinagm
            R[l][(2 * l + 1) * (-mp + l) + m + l] = d1 * sinag + d2 * sinagm
            R[l][(2 * l + 1) * (-mp + l) - m + l] = d1 * cosag - d2 * cosagm
            aux = cosmga * c3 - sinmga * s3
            sinmga = sinmga * c3 + cosmga * s3
            cosmga = aux
        sign *= -1
        aux = cosmal * c1 - sinmal * s1
        sinmal = sinmal * c1 + cosmal * s1
        cosmal = aux


def R(
    ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1, tol=1e-12, return_D=False
):

    c1 = cos_alpha
    s1 = sin_alpha
    c3 = cos_gamma
    s3 = sin_gamma

    root_two = np.sqrt((2.0))
    D = [np.nan * np.ones(((2 * l + 1) ** 2, 2 * l + 1)) for l in range(ydeg + 1)]
    R = [np.nan * np.ones(((2 * l + 1) ** 2, 2 * l + 1)) for l in range(ydeg + 1)]

    # Compute the initial matrices D0, R0, D1 and R1
    D[0][0] = [1]
    R[0][0] = [1]
    D[1][8] = [0, 0, 1]  # cos(phi/2)^2
    D[1][7] = [0, -root_two, 0]  # -sqrt(2) sin(phi/2) cos(phi/2)
    D[1][6] = [1, 0, 0]  # sin(phi/2)^2
    D[1][5] = -D[1][7]
    D[1][4] = D[1][8] - D[1][6]
    D[1][3] = D[1][7]
    D[1][2] = D[1][6]
    D[1][1] = D[1][5]
    D[1][0] = D[1][8]
    cosag = c1 * c3 - s1 * s3
    cosamg = c1 * c3 + s1 * s3
    sinag = s1 * c3 + c1 * s3
    sinamg = s1 * c3 - c1 * s3
    R[1][4] = D[1][4]
    R[1][7] = root_two * D[1][5] * c1
    R[1][1] = root_two * D[1][5] * s1
    R[1][5] = root_two * D[1][7] * c3
    R[1][3] = -root_two * D[1][7] * s3
    R[1][8] = D[1][8] * cosag - D[1][6] * cosamg
    R[1][6] = -D[1][8] * sinag - D[1][6] * sinamg
    R[1][2] = D[1][8] * sinag - D[1][6] * sinamg
    R[1][0] = D[1][8] * cosag + D[1][6] * cosamg

    # The remaining matrices are calculated using
    # symmetry and and recurrence relations
    for l in range(2, ydeg + 1):
        dlmn(l, s1, c1, s3, c3, D, R)

    if return_D:
        return R, D
    else:
        return R


def matprod(x1, x2):

    l1 = (x1.shape[0] - 1) // 2
    l2 = (x2.shape[0] - 1) // 2
    l = l1 + l2
    x1x2 = np.zeros((x1.shape[1], x2.shape[2], 2 * l + 1))
    for m, x1_m in enumerate(x1):
        for n, x2_n in enumerate(x2):
            x1x2[:, :, m + n] += np.dot(x1_m, x2_n)

    return x1x2


def MatrixDot(R, S):
    """
    Computes the dot product R . S . R^T.
    
    """
    N = S.shape[0]
    ydeg = int(np.sqrt(N) - 1)

    Ri = [None for _ in R]
    Rj = [None for _ in R]
    for l in range(ydeg + 1):
        Ri[l] = (R[l].T).reshape(2 * l + 1, 2 * l + 1, 2 * l + 1)
        Rj[l] = (R[l].reshape(2 * l + 1, 2 * l + 1, 2 * l + 1)).T

    RSRT = np.zeros((N, N, 4 * ydeg + 1))
    for l1 in range(ydeg + 1):
        i = slice(l1 ** 2, (l1 + 1) ** 2)
        for l2 in range(l1 + 1):
            j = slice(l2 ** 2, (l2 + 1) ** 2)
            K = 2 * (l1 + l2) + 1
            RiS = Ri[l1] @ S[i, j]
            RSRT[i, j, :K] = matprod(RiS, Rj[l2])
    return RSRT


def prod2(x1, x2):
    x1 = np.reshape(x1, (-1, x1.shape[-1]))
    x2 = np.reshape(x2, (-1, x2.shape[-1]))
    result = np.zeros(x1.shape[-1] + x2.shape[-1] - 1)
    for x1_n, x2_n in zip(x1, x2):
        X = np.outer(x1_n, x2_n)[::-1]
        for n, k in enumerate(range(-X.shape[0] + 1, X.shape[1])):
            result[n] += np.diag(X, k).sum()
    return result


def MatrixDotExplicit(R, S):
    """
    Computes the dot product R . S . R^T.
    
    """
    N = S.shape[0]
    ydeg = int(np.sqrt(N) - 1)

    Ri = [None for _ in R]
    Rj = [None for _ in R]
    for l in range(ydeg + 1):
        Ri[l] = (R[l].T).reshape(2 * l + 1, 2 * l + 1, 2 * l + 1)  # kji
        Rj[l] = R[l].reshape(2 * l + 1, 2 * l + 1, 2 * l + 1)  # ijk

    RSRT = np.zeros((N, N, 4 * ydeg + 1))
    for l1 in range(ydeg + 1):
        i = slice(l1 ** 2, (l1 + 1) ** 2)
        for l2 in range(l1 + 1):
            j = slice(l2 ** 2, (l2 + 1) ** 2)
            K = 2 * (l1 + l2) + 1
            RiS = (Ri[l1] @ S[i, j]).T  # kji -> ijk
            for ii in range(2 * l1 + 1):
                for jj in range(2 * l2 + 1):
                    for kk in range(2 * l2 + 1):
                        RSRT[l1 * l1 + ii, l2 * l2 + jj, :K] += prod(
                            RiS[kk, ii], Rj[l2][jj, kk]
                        )

    return RSRT


# DEBUG
ydeg = 10
N = (ydeg + 1) ** 2
S = np.random.randn(N, N) / N
S += S.T
S = np.eye(N)

pyR = wigner.R(ydeg)
tstart = time.time()
res1 = wigner.MatrixDot(pyR, S)
print(time.time() - tstart)

tstart = time.time()
res2 = starry._c_ops.gp(ydeg, S)
print((time.time() - tstart) / 11)
