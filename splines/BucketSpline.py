import numpy as np
import csaps


class BucketSpline(object):

    def __init__(self, x, z, y_buckets=10, smooth=None, normalizedsmooth=False):
        order = np.argsort(x[:, 0])
        d = x.shape[1]
        x = x[order]
        z = z[order]
        if y_buckets > 1 and d > 1:
            self.buckets = np.quantile(x[:, 1], np.linspace(1/y_buckets, 1, y_buckets)[:-1])
            b = np.digitize(x[:, 1], self.buckets)
        else:
            self.buckets = np.zeros(0)
            b = np.zeros(x.shape[0], dtype=int)
            y_buckets = 1
        # (unique, counts) = np.unique(b, return_counts=True)

        #

        self.splines = []
        self.residual = 0
        for i in range(y_buckets):
            xb = x[b == i, 0].flatten()
            if xb.size < 3:
                i = i+1
                xb = x[b == i, 0].flatten()
            zb = z[b == i]
            #smooth = self.calc_smooth(xb, 0.5)
            spline = csaps.csaps(xb, zb, smooth=smooth, normalizedsmooth=False)
            self.splines.append(spline)
            v = spline(xb)
            self.residual = self.residual + np.sum((zb-v)**2)

    def calc_smooth(self, x, smooth):

        span = np.ptp(x)

        eff_x = 1 + (span ** 2) / np.sum(np.diff(x) ** 2)
        eff_w = x.shape[0]
        k = 80 * (span ** 3) * (x.size ** -2) * (eff_x ** -0.5) * (eff_w ** -0.5)

        s = 0.5 if smooth is None else smooth
        p = s / (s + (1 - s) * k)

        return max(0.0000000001, min(0.99999, p))

    def __call__(self, X, dx=0, dy=0, daxis=None):
        d = X.shape[1]
        x = X[:, 0]
        n = X.shape[0]
        if d > 1 and self.buckets.size > 0:
            b = np.digitize(X[:, 1], self.buckets)
        else:
            b = np.zeros(n, dtype=int)
        if not daxis is None:
            if daxis == 0:
                dx = 1
            elif daxis == 1:
                dy = 1
            else:
                raise ValueError("invalid daxis")
        result = np.empty(x.size)
        for i in range(x.size):
            v = self.splines[b[i]](x[i], nu=dx)
            if dy > 0:
                if 0 < b[i] < self.buckets.size-1:
                    left = self.splines[b[i]-1](x[i], nu=dx)
                    right = self.splines[b[i]+1](x[i], nu=dx)
                    result[i] = 0.5*(right-v)/(self.buckets[b[i]+1]-self.buckets[b[i]]) + 0.5*(v-left)/(self.buckets[b[i]]-self.buckets[b[i]-1])
                elif b[i] > 0:  # b[i] >= self.buckets.size-1
                    left = self.splines[b[i] - 1](x[i], nu=dx)
                    result[i] = (v - left) / (self.buckets[-1] - self.buckets[-2])
                elif self.buckets.size > 0:  # b[i] == 0
                    right = self.splines[b[i] + 1](x[i], nu=dx)
                    result[i] = (right-v)/(self.buckets[b[i]+1]-self.buckets[b[i]])
                else:
                    result[i] = 0
            else:
                result[i] = v

        return result

    def get_residual(self):
        return self.residual

    def get_coeffs(self):
        n = self.splines[0].spline.coeffs.shape[0]
        m_all = np.array([s.spline.coeffs.shape[1] for s in self.splines])
        m = np.max(m_all)
        c = [np.pad(s.spline.coeffs,((0,n-s.spline.coeffs.shape[0]),(0,m-s.spline.coeffs.shape[1]))) for s in self.splines]
        return np.stack(c)



