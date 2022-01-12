import numpy as np

from mckeangenerator import IPathGenerator


class IPayout(object):

    def __init__(self):
        self.name = "unknown"
        self.h = None

    def __call__(self, x, u=None):
        """
        :param x:
        :param u: if None independent series will be generated, if needed
        :return:
        """
        pass


class CallPayout(IPayout):

    def __init__(self, strike, maturity_idx=-1):
        self.name = "call"
        self.strike = strike
        self.maturity_idx = maturity_idx

    def __call__(self, x, u=None):
        return np.fmax(x[:, self.maturity_idx, 0]-self.strike, 0)

    def gradient(self, x):
        result = np.zeros((x.shape[0], x.shape[2]))
        result[:, 0] = (x[:, self.maturity_idx, 0] >= self.strike).astype(x.dtype)
        return result


class SquaredPayout(IPayout):

    def __init__(self, maturity_idx=-1):
        self.name = "squared"
        self.maturity_idx = maturity_idx

    def __call__(self, x, u=None):
        return x[:, self.maturity_idx, 0]**2

    def gradient(self, x):
        result = np.zeros((x.shape[0], x.shape[2]))
        result[:, 0] = 2*x[:, self.maturity_idx, 0]
        return result


class TrigonometricPayout(IPayout):

    def __init__(self, generator: IPathGenerator, maturity_idx=-1):
        self.name = "trigonometric"
        self.generator = generator
        self.maturity_idx = maturity_idx

    def __call__(self, x, u=None):
        if self.h is None:
            raise ValueError("property h needs to be set")
        N = x.shape[0]
        if u is None:
            L = x.shape[1]
            u, _ = self.generator.generate(N, L, self.h)
        xv = x[:, self.maturity_idx, 0].reshape((N, 1))
        uv = u[:, self.maturity_idx, 0].reshape((N, 1)).transpose()
        return np.mean(np.cos(xv) * np.sin(uv), axis=1)