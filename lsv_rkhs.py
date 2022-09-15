import warnings

import numpy as np
import scipy.interpolate
import marketvol
from mckeangenerator import IPathGenerator
from splines.SmoothSpline import SmoothSpline
import rkhs.finance


class LsvGenerator(IPathGenerator):
    def __init__(self, r=0, kappa=9, v0=0.16*0.16, theta=0.16*0.16, xi=0.4, rho=-0.5, market_vol=marketvol.MarketVol.load_csv()):
        self.v_lower_bound = 0.01*0.01
        v_level = theta-xi**2/(2*kappa)
        print("Lower variance bound set to ", self.v_lower_bound)
        self.name = "lsvrkhs"
        self.market_vol = market_vol
        print("Current price s0 ",  self.market_vol.s0)
        self.r = r
        self.kappa = kappa
        self.v0 = v0
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.lambd = 0.0001
        self.varkernel = 5
        self.Nkernels = 16
        self.withconstant = True

    def drift(self, X, t):
        x = X[:, 0]
        v = X[:, 1]
        n = x.shape[0]
        result = np.empty((n, 2))
        result[:, 0] = self.r * x
        result[:, 1] = self.kappa * (self.theta - v)
        return result

    def brownian_cov(self, X, t):
        n = X.shape[0]
        result = np.empty((n, 2, 2))
        result[:, 0, 0] = 1
        result[:, 0, 1] = 0
        result[:, 1, 0] = self.rho
        result[:, 1, 1] = np.sqrt(1 - self.rho ** 2)
        return result

    def diffusion(self, X, t):
        x = X[:, 0]
        v = X[:, 1]
        n = x.shape[0]
        dupire = np.zeros(n)
        for i in range(n):
            dupire[i] = self.market_vol.dupire_vol(t, x[i])
        result = np.empty((n, 2, 2))

        eps = self.v_lower_bound
        fsqY = np.maximum(v, eps)  # f^2(Y)
        estimatedrift, estimatediffusion = rkhs.finance.mlambdaKRRfast(x, v, fsqY, self.lambd, self.varkernel, self.Nkernels, self.withconstant)

        #driftpart = driftX(curtime, prevX) * h(curtime, prevY) / (np.maximum(eps, estimatedrift))
        diffusionpart = x * dupire * np.sqrt(np.maximum(v, eps)) / (np.sqrt(np.maximum(eps, estimatediffusion)))
        #resultX = prevX + driftpart * delta + diffusionpart * curnoiseX * sqrtdelta
        #resultY = prevY + driftY(curtime, prevY) * delta + sigma2(curtime, prevY) * curnoiseY * sqrtdelta
        diff_v = self.xi * np.sqrt(np.maximum(v, eps))
        if np.max(diffusionpart) > 100000 or np.min(diffusionpart) < 0:
            warnings.warn("outlier value")
        result[:, 0, 0] = diffusionpart
        result[:, 0, 1] = 0
        result[:, 1, 0] = diff_v * self.rho
        result[:, 1, 1] = diff_v * np.sqrt(1 - self.rho ** 2)
        return result

    def calc_values(self, dW, h):
        N = dW.shape[0]
        L = dW.shape[1]-1
        x0 = self.market_vol.s0
        x = np.zeros((N, L + 1, 2))
        x[:, 0, 0] = x0
        x[:, 0, 1] = self.v0
        for l in range(1, L + 1):
            t = (l - 1) * h
            drift = self.drift(x[:, l - 1, :], t)
            diffusion = self.diffusion(x[:, l - 1, :], t)
            x[:, l, 0] = x[:, l - 1, 0] + drift[:, 0] * h + diffusion[:, 0, 0] * dW[:, l - 1, 0] + diffusion[:, 0, 1] * dW[:, l - 1, 1]
            x[:, l, 1] = np.fmax(self.v_lower_bound, x[:, l - 1, 1] + drift[:, 1] * h + diffusion[:, 1, 0] * dW[:, l - 1, 0] + diffusion[:, 1, 1] * dW[:, l - 1, 1])
        return x

    def generate(self, N, L, h, return_diffusion_delta=False):
        x0 = self.market_vol.s0
        x = np.zeros((N, L + 1, 2))
        x[:, 0, 0] = x0
        x[:, 0, 1] = self.v0
        dW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1, 2))
        B = np.empty((N, L+1, 2))
        for l in range(1, L + 1):
            t = (l-1)*h
            drift = self.drift(x[:, l-1, :], t)
            diffusion = self.diffusion(x[:, l-1, :], t)
            x[:, l, 0] = x[:, l - 1, 0] + drift[:, 0] * h + diffusion[:, 0, 0] * dW[:, l - 1, 0] + diffusion[:, 0, 1] * dW[:, l - 1, 1]
            x[:, l, 1] = np.fmax(self.v_lower_bound, x[:, l - 1, 1] + drift[:, 1] * h + diffusion[:, 1, 0] * dW[:, l - 1, 0] + diffusion[:, 1, 1] * dW[:, l - 1, 1])
            B[:, l-1, 0] = diffusion[:, 0, 0] * dW[:, l-1, 0] + diffusion[:, 0, 1] * dW[:, l-1, 1]
            B[:, l-1, 1] = diffusion[:, 1, 0] * dW[:, l-1, 0] + diffusion[:, 1, 1] * dW[:, l-1, 1]
        if return_diffusion_delta:
            diffusion = self.diffusion(x[:, -1, :], t)
            B[:, -1, 0] = diffusion[:, 0, 0] * dW[:, -1, 0] + diffusion[:, 0, 1] * dW[:, -1, 1]
            B[:, -1, 1] = diffusion[:, 1, 0] * dW[:, -1, 0] + diffusion[:, 1, 1] * dW[:, -1, 1]
            test = np.empty((N, L + 1, 2))
            test[:, :, 0] = dW[:, :, 0]
            test[:, :, 1] = dW[:, :, 0] * self.rho + dW[:, :, 1] * np.sqrt(1 - self.rho ** 2)
            return x, dW, test
        return x, dW

    def generate_localvol(self, N, L, h):
        x0 = self.market_vol.s0
        x = np.zeros((N, L + 1))
        x[:, 0] = x0
        dW = np.math.sqrt(h) * np.random.normal(0, 1, (N, L + 1))
        for l in range(1, L + 1):
            t = (l-1)*h
            for i in range(N):
                x[i, l] = x[i, l - 1] + x[i, l-1] * self.r * h + x[i, l-1] * self.market_vol.dupire_vol(t, x[i, l - 1]) * dW[i, l - 1]
        return x, dW

    def get_dimensions(self):
        return 2, 2

    def get_complexity(self, N):
        return N*self.Nkernels*self.Nkernels

