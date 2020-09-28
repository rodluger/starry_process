import starry
import matplotlib.pyplot as plt
import numpy as np

# Settings
ninc = 10
ntheta = 100
ydeg = 5

# Construct a design matrix for a star observed
# over the full range of inclinations
map = starry.Map(ydeg, lazy=False)
incs = np.array(np.arange(10, 91, ninc), dtype=float)
theta = np.linspace(0, 360, ntheta, endpoint=False)
A = np.empty((0, map.Ny))
for inc in incs:
    map.inc = inc
    # This design matrix
    Ai = map.design_matrix(theta=theta)
    # Cumulative design matrix
    A = np.vstack((A, Ai))


# Get the null space, Q
U, S, VT = np.linalg.svd(A)
k = np.linalg.rank(A)
Q = VT[k:].T


# TODO
