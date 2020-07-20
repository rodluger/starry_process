import numpy as np

np.random.seed(0)


X3 = np.random.randint(0, 50, (2, 3, 5))


X2 = X3.reshape(2 * 3, 5)


Y3 = X3.T


Y2 = Y3.reshape(5 * 3, 2)


print(X2)

print(Y2)


print(np.vstack([X2[:, k].reshape(2, 3).T for k in range(5)]))
