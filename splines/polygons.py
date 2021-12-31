import numpy as np


def hermite_poly(x, K):
    A = np.ones((x.shape[0], K))
    A[:, 1] = x
    for i in range(2, K):
      A[:, i] = x * A[:, i-1] - (i-1) * A[:, i-2]
    return A


def hermite_deriv_poly(x, K):
    base = hermite_poly(x, K)
    result = np.zeros((x.shape[0], K))
    for i in range(1, K):
        result[:, i] = np.sqrt(2) * i * base[:, i-1]
    return result
