import numpy as np
from scipy.interpolate import bisplrep, bisplev


class GridSpline(object):
    def __init__(self, x, z, x_knots=8, y_knots=8, smooth=None):
        self.z_norm = np.std(z)
        z = z / self.z_norm
        self.x_knots = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), x_knots)
        self.y_knots = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), y_knots)
        n = x.shape[0]
        #m = n/np.std(z)
        smooth = n+np.sqrt(2*n)
        self.tck, self.fp, ier, msg = bisplrep(x[:, 0].flatten(), x[:, 1].flatten(), z.flatten(), task=0, s=smooth, tx=self.x_knots, ty=self.y_knots, full_output=True)#, kx=3, ky=3, task=-1, s=smooth)
        print(msg)

    def __call__(self, X, dx=0, dy=0, daxis=None):
        if not daxis is None:
            if daxis == 0:
                dx = 1
            elif daxis == 1:
                dy = 1
            else:
                raise ValueError("invalid daxis")
        result = np.empty((X.shape[0]))
        for i in range(result.size):
            result[i] = self.z_norm*bisplev(X[i, 0].flatten(), X[i, 1].flatten(), self.tck, dx=dx, dy=dy)
        return result

    def get_residual(self):
        return self.fp

    def get_coeffs(self):
        return self.tck[2]

