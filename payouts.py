import numpy as np

from mckeangenerator import IPathGenerator


class IPayout(object):

    def __init__(self):
        self.name = "unknown"

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


class TrigonometricPayout(IPayout):

    def __init__(self, h, generator: IPathGenerator, maturity_idx=-1):
        self.name = "trigonometric"
        self.generator = generator
        self.maturity_idx = maturity_idx
        self.h = h

    def __call__(self, x, u=None):
        N = x.shape[0]
        if u is None:
            L = x.shape[1]
            u, _ = self.generator.generate(N, L, self.h)
        xv = x[:, self.maturity_idx, 0].reshape((N, 1))
        uv = u[:, self.maturity_idx, 0].reshape((N, 1)).transpose()
        return np.mean(np.cos(xv) * np.sin(uv), axis=1)