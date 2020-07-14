import numpy as np
import matplotlib.pyplot as plt
from starry_process.integrals import wigner
import starry
import time


ydeg = 10
N = (ydeg + 1) ** 2
S = np.random.randn(N, N) / N
S += S.T

# python
pyR = wigner.R(ydeg)
tstart = time.time()
res1 = wigner.MatrixDot(pyR, S)
print(time.time() - tstart)

# C++
tstart = time.time()
res2 = starry._c_ops.gp(ydeg, S)
print((time.time() - tstart) / 11)

# Format for comparison
res2 = res2.reshape(N, N, 4 * ydeg + 1)
for k in range(4 * ydeg + 1):
    res1[:, :, k] = np.tril(res1[:, :, k])
    res2[:, :, k] = np.tril(res2[:, :, k])

print(np.allclose(res1, res2))
breakpoint()
