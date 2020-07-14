import numpy as np


__all__ = ["R"]


def prod(x1, x2):
    l1 = (len(x1) - 1) // 2
    l2 = (len(x2) - 1) // 2
    l = l1 + l2
    result = np.zeros(2 * l + 1)
    for m, x1_m in enumerate(x1):
        for n, x2_n in enumerate(x2):
            result[m + n] += x1_m * x2_n
    return result


def matprod(x1, x2):
    l1 = (x1.shape[0] - 1) // 2
    l2 = (x2.shape[0] - 1) // 2
    l = l1 + l2
    result = np.zeros((2 * l + 1, x1.shape[1], x2.shape[2]))
    for m, x1_m in enumerate(x1):
        for n, x2_n in enumerate(x2):
            result[m + n] += np.dot(x1_m, x2_n)
    return result


def shift_left(x):
    return np.append(x[1:], [0])


def dlmn(l, s1, c1, s3, c3, D, R):

    iinf = 1 - l
    isup = -iinf

    # Compute the D[lm',m) matrix.
    # First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    D[l][2 * l, 2 * l] = prod(D[l - 1][isup + l - 1, isup + l - 1], [0, 0, 1])
    D[l][2 * l, 0] = prod(D[l - 1][isup + l - 1, -isup + l - 1], [1, 0, 0])
    for m in range(isup, iinf - 1, -1):
        # Multiplication by s/c
        D[l][2 * l, m + l] = shift_left(
            -np.sqrt((l + m + 1.0) / (l - m)) * D[l][2 * l, m + 1 + l]
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
            D[l][mp + l, m + l] = prod(
                fact * (2 * l - 1) * a * D[l - 1][mp + l - 1, m + l - 1],
                [b - 1, 0, b + 1],
            )
            if (lbuz != 1) and (lbux != 1):
                # Trick: Promote D[l - 2] to D[l] by multiplying by (c^2 + s^2)^2
                cuz = np.sqrt(((lauz - 1) * (lbuz - 1)))
                D[l][mp + l, m + l] -= (fact * cux * cuz) * prod(
                    D[l - 2][mp + l - 2, m + l - 2], [1, 0, 2, 0, 1]
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
            D[l][mp + l, m + l] = sign * D[l][m + l, mp + l]
            sign *= -1
        iinf += 1
        isup -= 1

    # Inversion
    iinf = -l
    isup = iinf
    for m in range(l - 1, -(l + 1), -1):
        sign = -1
        for mp in range(isup, iinf - 1, -1):
            D[l][mp + l, m + l] = sign * D[l][-mp + l, -m + l]
            sign *= -1
        isup += 1

    # Compute the real rotation matrices R from the complex ones D
    R[l][l, l] = D[l][l, l]
    cosmal = c1
    sinmal = s1
    sign = -1
    root_two = np.sqrt((2.0))
    for mp in range(1, l + 1):
        cosmga = c3
        sinmga = s3
        aux = root_two * D[l][0 + l, mp + l]
        R[l][mp + l, 0 + l] = aux * cosmal
        R[l][-mp + l, 0 + l] = aux * sinmal
        for m in range(1, l + 1):
            aux = root_two * D[l][m + l, 0 + l]
            R[l][l, m + l] = aux * cosmga
            R[l][l, -m + l] = -aux * sinmga
            d1 = D[l][-mp + l, -m + l]
            d2 = sign * D[l][mp + l, -m + l]
            cosag = cosmal * cosmga - sinmal * sinmga
            cosagm = cosmal * cosmga + sinmal * sinmga
            sinag = sinmal * cosmga + cosmal * sinmga
            sinagm = sinmal * cosmga - cosmal * sinmga
            R[l][mp + l, m + l] = d1 * cosag + d2 * cosagm
            R[l][mp + l, -m + l] = -d1 * sinag + d2 * sinagm
            R[l][-mp + l, m + l] = d1 * sinag + d2 * sinagm
            R[l][-mp + l, -m + l] = d1 * cosag - d2 * cosagm
            aux = cosmga * c3 - sinmga * s3
            sinmga = sinmga * c3 + cosmga * s3
            cosmga = aux
        sign *= -1
        aux = cosmal * c1 - sinmal * s1
        sinmal = sinmal * c1 + cosmal * s1
        cosmal = aux


def R(ydeg, cos_alpha=0, sin_alpha=1, cos_gamma=0, sin_gamma=-1, tol=1e-12):

    c1 = cos_alpha
    s1 = sin_alpha
    c3 = cos_gamma
    s3 = sin_gamma

    root_two = np.sqrt((2.0))
    D = [np.nan * np.ones((2 * l + 1, 2 * l + 1, 2 * l + 1)) for l in range(ydeg + 1)]
    R = [np.nan * np.ones((2 * l + 1, 2 * l + 1, 2 * l + 1)) for l in range(ydeg + 1)]

    # Compute the initial matrices D0, R0, D1 and R1
    D[0][0, 0] = [1]
    R[0][0, 0] = [1]
    D[1][2, 2] = [0, 0, 1]  # cos(phi/2)^2
    D[1][2, 1] = [0, -root_two, 0]  # -sqrt(2) sin(phi/2) cos(phi/2)
    D[1][2, 0] = [1, 0, 0]  # sin(phi/2)^2
    D[1][1, 2] = -D[1][2, 1]
    D[1][1, 1] = D[1][2, 2] - D[1][2, 0]
    D[1][1, 0] = D[1][2, 1]
    D[1][0, 2] = D[1][2, 0]
    D[1][0, 1] = D[1][1, 2]
    D[1][0, 0] = D[1][2, 2]
    cosag = c1 * c3 - s1 * s3
    cosamg = c1 * c3 + s1 * s3
    sinag = s1 * c3 + c1 * s3
    sinamg = s1 * c3 - c1 * s3
    R[1][1, 1] = D[1][1, 1]
    R[1][2, 1] = root_two * D[1][1, 2] * c1
    R[1][0, 1] = root_two * D[1][1, 2] * s1
    R[1][1, 2] = root_two * D[1][2, 1] * c3
    R[1][1, 0] = -root_two * D[1][2, 1] * s3
    R[1][2, 2] = D[1][2, 2] * cosag - D[1][2, 0] * cosamg
    R[1][2, 0] = -D[1][2, 2] * sinag - D[1][2, 0] * sinamg
    R[1][0, 2] = D[1][2, 2] * sinag - D[1][2, 0] * sinamg
    R[1][0, 0] = D[1][2, 2] * cosag + D[1][2, 0] * cosamg

    # The remaining matrices are calculated using
    # symmetry and and recurrence relations
    for l in range(2, ydeg + 1):
        dlmn(l, s1, c1, s3, c3, D, R)

    return R
