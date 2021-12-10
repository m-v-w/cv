import numpy as np
import scipy.interpolate
import marketvol


class LsvGenerator(object):
    def __init__(self, r, kappa, v0, theta, xi, rho):
        self.v_lower_bound = theta-xi**2/(2*kappa)
        print("Lower variance bound", self.v_lower_bound)
        self.market_vol = marketvol.MarketVol.load_csv()
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def drift(self, X, t):
        d1 = self.r * X[:, 0]
        d2 = self.kappa * (self.theta - X[:, 1])
        return np.stack((d1, d2), axis=1)

    def diffusion(self, X, t):
        x = X[:,0]
        v = X[:,1]
        n = x.shape[0]
        cond_v = np.ones(n)*self.v0
        if t > 0:
            order = np.argsort(x)
            #csaps(x[order], v[order], smooth=0.9)
            spline = scipy.interpolate.UnivariateSpline(x[order], v[order])
            for i in range(n):
                cond_v[i] = np.fmax(self.v_lower_bound, spline(x[i]))
        dupire = np.zeros(n)
        for i in range(n):
            dupire[i] = self.market_vol.dupire_vol(t, x[i])
        d1 = np.zeros((n, 2))
        d1[:, 0] = dupire / np.sqrt(cond_v) * x * np.sqrt(v)
        d21 = self.xi * np.sqrt(v) * self.rho
        d22 = self.xi * np.sqrt(v) * np.sqrt(1 - self.rho ** 2)
        d2 = np.stack((d21, d22), axis=1)
        return np.stack((d1, d2), axis=1)

    def generate(self, N, L, h):
        x0 = self.market_vol.s0
        x = np.zeros((N, L + 1, 2))
        x[:, 0, 0] = x0
        x[:, 0, 1] = self.v0
        dW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1, 2))
        for l in range(1, L + 1):
            t = (l-1)*h
            drift = self.drift(x[:, l-1, :], t)
            diffusion = self.diffusion(x[:, l-1, :], t)
            x[:, l, 0] = x[:, l - 1, 0] + drift[:, 0] * h + diffusion[:, 0, 0] * dW[:, l - 1, 0] + diffusion[:, 0, 1] * dW[:, l - 1, 1]
            x[:, l, 1] = np.fmax(self.v_lower_bound, x[:, l - 1, 1] + drift[:, 1] * h + diffusion[:, 1, 0] * dW[:, l - 1, 0] + diffusion[:, 1, 1] * dW[:, l - 1, 1])
        return x, dW
