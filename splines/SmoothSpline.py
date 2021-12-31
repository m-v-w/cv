import numpy as np
from scipy.interpolate import splrep, splev


class SmoothSpline(object):
    def __init__(self, x, z, x_knots=8, smooth=None):
        self.z_norm = np.std(z)
        z = z / self.z_norm
        self.x_knots = np.linspace(np.min(x), np.max(x), x_knots)
        n = x.shape[0]
        #m = n/np.std(z)
        smooth = n+np.sqrt(2*n)
        order = np.argsort(x)
        self.tck, self.fp, ier, msg = splrep(x[order], z[order], task=0, s=smooth, full_output=True)  # , t=self.x_knots
        if ier > 0:
            print(msg)

    def __call__(self, X, dx=0, daxis=None):
        if not daxis is None:
            if daxis == 0:
                dx = 1
            else:
                raise ValueError("invalid daxis")
        result = self.z_norm*splev(X, self.tck, der=dx)
        return result

    def get_residual(self):
        return self.fp

    def get_coeffs(self):
        return self.tck[2]
