import numpy as np


class IPathGenerator(object):

    def drift(self, X, t):
        pass

    def diffusion(self, X, t):
        """
        :param X: [sample, dimension_index]
        :param t: time
        :return: [sample, dimension_index, brownian_motion_index]
        """
        pass

    def generate(self, N, L, h):
        pass

    def get_dimensions(self):
        """
        :return: X-dimensions, dW-dimensions
        """
        pass

    def generate_diffusions(self, x, dw, h):
        """

        :param x: [sample, dimension_index]
        :param dw: [sample, brownian_motion_index]
        :param h: time step
        :return: [sample, time_index, dimension_index]
        """
        N = x.shape[0]
        L = x.shape[1] - 1
        d_x = x.shape[2]
        d_w = dw.shape[2]
        B = np.zeros((N, L, d_x))
        for l in range(L):
            xm = np.tile(x[:, l], (N, 1))
            t = (l - 1) * h
            diffusion = self.diffusion(x[:, l, :], t)
            for k_x in range(d_x):
                for k_w in range(d_w):
                    B[:, l, k_x] = B[:, l, k_x] + diffusion[:, k_x, k_w] * dw[:, l + 1, k_w]
        return B


class SimpleGenerator(IPathGenerator):

    def drift(self, X, t):
        N = X.shape[0]
        x = X
        u_square = X.T ** 2
        result = (1 + u_square) * np.exp(-u_square / 2) + x * (-1) * np.exp(-u_square / 2)
        # result = (1 + np.square(u)) * np.exp(-np.square(u) / 2) + x * (-1) * np.exp(-np.square(u) / 2)
        return np.mean(result, axis=1)

    def diffusion(self, X, t):
        n = X.shape[0]
        return np.ones((n, 1, 1))

    def get_dimensions(self):
        return 1, 1

    def generate(self, N, L, h):
        x0 = 0
        x = np.zeros((N, L + 1, 1))
        x[:, 0, 0] = x0
        dW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1, 1))
        for l in range(1, L + 1):
            t = (l - 1) * h
            drift = self.drift(x[:, l - 1, :], t)
            x[:, l, 0] = x[:, l - 1, 0] + drift * h + dW[:, l - 1, 0]
        return x, dW


class SimpleCorrGenerator(IPathGenerator):

    def __init__(self, rho):
        self.rho = rho

    def drift(self, X, t):
        n = X.shape[0]
        result = np.empty((n, 2))
        for k in range(2):
            x = np.reshape(X[:, k], (n, 1))
            u_square = x.T ** 2
            result[:, k] = np.mean((1 + u_square) * np.exp(-u_square / 2) + x * (-1) * np.exp(-u_square / 2), axis=1)
        return result

    def diffusion(self, X, t):
        n = X.shape[0]
        result = np.empty((n, 2, 2))
        result[:, 0, 0] = self.rho
        result[:, 0, 1] = np.sqrt(1 - self.rho ** 2)
        result[:, 1, 0] = 1
        result[:, 1, 1] = 0
        return result

    def bMat(self, x, dw):
        N = x.shape[0]
        L = x.shape[1] - 1
        d=2
        B = np.zeros((N, L, 2))
        for l in range(L):
            xm = np.tile(x[:, l], (N, 1))
            B[:, l] = np.mean(b(xm.transpose(), xm), 1) * dw[:, l + 1]
        return B


    def get_dimensions(self):
        return 2, 2

    def generate(self, N, L, h):
        x = np.zeros((N, L + 1, 2))
        dW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1, 2))
        for l in range(1, L + 1):
            t = (l - 1) * h
            drift = self.drift(x[:, l - 1, :], t)
            diffusion = self.diffusion(x[:, l - 1, :], t)
            x[:, l, 0] = x[:, l - 1, 0] + drift[:, 0] * h + diffusion[:, 0, 0] * dW[:, l - 1, 0] + diffusion[:, 0, 1] * dW[:, l - 1, 1]
            x[:, l, 1] = x[:, l - 1, 1] + drift[:, 1] * h + diffusion[:, 1, 0] * dW[:, l - 1, 0] + diffusion[:, 1, 1] * dW[:, l - 1, 1]
        return x, dW
