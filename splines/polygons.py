import numpy as np


def hermite_poly_multi(X, K):
    d = X.shape[1]
    len = K-1
    result = np.zeros((X.shape[0], d*len+1))
    result[:, 0] = 1
    for m in range(d):
        result[:, 1+m*len:1+(m+1)*len] = hermite_poly(X[:, m], K)[:, 1:]
    return result


def hermite_poly(x, K):
    A = np.ones((x.shape[0], K))
    A[:, 1] = x
    for i in range(2, K):
      A[:, i] = x * A[:, i-1] - (i-1) * A[:, i-2]
    return A


def hermite_poly_eval_deriv(X, K, c):
    d = X.shape[1]
    result = np.empty((X.shape[0], d))
    len = K - 1
    for m in range(d):
        A = hermite_deriv_poly(X[:, m], K)
        fit = np.zeros(K)
        fit[1:] = c[1+m*len:1+(m+1)*len]
        result[:, m] = A @ fit
    return result


def hermite_deriv_poly(x, K):
    base = hermite_poly(x, K)
    result = np.zeros((x.shape[0], K))
    for i in range(1, K):
        result[:, i] = np.sqrt(2) * i * base[:, i-1]
    return result
