import numpy as np

def genproc_1dim_ex(L, h, N, a, b):
    x = np.zeros((N, L + 1))
    deltaW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1))
    for l in range(1, L + 1):
        for i in range(N):
            x[i, l] = x[i, l - 1] + np.mean(a(np.ones(N) * x[i, l - 1], x[:, l - 1]) * h)\
                + np.mean(b(x[i, l - 1], x[:, l - 1]) * deltaW[i, l - 1])
    return x, deltaW


def a(x, u):
    return (1 + np.square(u)) * np.exp(-np.square(u) / 2) + x * (-1) * np.exp(-np.square(u) / 2)


def b(x, u):
    if np.isscalar(x):
        return np.ones(u.shape[0])
    else:
        return np.ones((x.shape[0], u.shape[0]))


def bMat(x, dw):
    N = x.shape[0]
    L = x.shape[1] - 1
    B = np.zeros((N, L))
    for l in range(L):
        xm = np.tile(x[:, l], (N, 1))
        B[:, l] = np.mean(b(xm.transpose(), xm), 1) * dw[:, l + 1]
    return B


def f_call(x, u):
    n = x.size
    y = np.fmax(x-0.5, 0)
    return np.tile(y, (n, 1)).transpose()


def f(x, u):
    #return np.fmax(x-u, 0)
    return np.cos(x)*np.sin(u)


def genPoly(x, K):
    n = x.shape[0]
    A = np.ones((n, K))
    for i in range(1, K-1):
        A[:, i] = x * A[:, i-1]
    A[:, K-1] = np.mean(f(x.reshape((n, 1))*np.ones((1, n)), np.ones((1, n))*x.reshape((n, 1))), 1)
    return A


def gen_baseMat(x, dW):
    N = x.shape[0]
    L = x.shape[1]-1
    K = 5
    B = np.zeros((N, L*K))
    for l in range(L):
        A = genPoly(x[:, l], K)
        bv = np.mean(b(x[:, l]*np.ones((1, N)), np.ones((N, 1))*x[:, l]), 2)
        B[:, ((l-1)*K+1):(l*K)] = A*(bv*dW[:, l+1]*np.ones((1,K)))

