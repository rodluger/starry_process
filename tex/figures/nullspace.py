import starry
import matplotlib.pyplot as plt
import numpy as np

# Settings
ninc = 3
ntheta = 100
ydeg = 5


def null_space(A):
    U, S, VT = np.linalg.svd(A)
    k = np.linalg.matrix_rank(A)
    return VT[k:].T


# Construct a design matrix for a star observed
# over the full range of inclinations
map = starry.Map(ydeg, lazy=False)
incs = np.linspace(10.0, 90.0, ninc)
theta = np.linspace(0, 360, ntheta, endpoint=False)

A = [None for i in range(ninc)]
Q = [None for i in range(ninc + 1)]

for i, inc in enumerate(incs):
    map.inc = inc
    A[i] = map.design_matrix(theta=theta)
    Q[i] = null_space(A[i])

Aall = np.vstack(A)
Q[-1] = null_space(Aall)

plt.switch_backend("MacOSX")
fig, ax = plt.subplots(ninc + 1)
for i, axis in enumerate(ax):
    axis.imshow(Q[i])
plt.show()

# breakpoint()
# pass
