import numpy as np


class IPayout(object):

    def __call__(self, x, u=None):
        """
        :param x:
        :param u: if None independent series will be generated, if needed
        :return:
        """
        pass


class CallPayout(IPayout):

    def __init__(self, strike, maturity_idx=-1):
        self.strike = strike
        self.maturity_idx = maturity_idx

    def __call__(self, x, u=None):
        return np.fmax(x[:, self.maturity_idx, 0]-self.strike, 0)
