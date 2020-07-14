from starry_gp.latitude import LatitudeIntegral
from wigner import R
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
from starry_process.integrals.latitude import LatitudeIntegral as LatitudeIntegralSlow

ydeg = 3

# Random vector
np.random.seed(0)
N = (ydeg + 1) ** 2
s = np.random.randn(N)
sqrtS = np.random.randn(N, N) / N
S = sqrtS @ sqrtS.T

# Get analytic integral
L = LatitudeIntegral(ydeg)
L.set_params(2.0, 3.0)
mu = L.first_moment(s)
C = L.second_moment(sqrtS)


# Get analytic integral (slow version)
L = LatitudeIntegralSlow(ydeg)
L.set_params(2.0, 3.0)
L.set_vector(s)
L.set_matrix(S)
mu_slow = L.integral1()
C_slow = L.integral2()

# Compare
plt.plot(mu, lw=3)
plt.plot(mu_slow, lw=1)

plt.figure()
plt.imshow(C)
plt.colorbar()

plt.figure()
plt.imshow(C_slow)
plt.colorbar()

plt.figure()
plt.imshow(C / C_slow)
plt.colorbar()

plt.show()
