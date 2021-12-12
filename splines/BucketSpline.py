import numpy as np
import csaps


class BucketSpline(object):

    def __init__(self, x, y, z, y_buckets=10, smooth=0.9):
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        z = z[order]
        self.buckets = np.quantile(y, np.linspace(1/y_buckets, 1, y_buckets)[:-1])
        b = np.digitize(y, self.buckets)
        (unique, counts) = np.unique(b, return_counts=True)

        self.splines = []
        for i in range(y_buckets):
            xb = x[b == i]
            if xb.size < 3:
                i = i+1
                xb = x[b == i]
            zb = z[b == i]
            spline = csaps.csaps(xb, zb, xb, smooth=smooth)
            self.splines.append(spline)

    def __call__(self, x, y, dx=0, dy=0):
        b = np.digitize(y, self.buckets)
        result = np.empty(x.size)
        for i in range(x.size):
            v = self.splines[b[i]](x, dx=dx)
            if dy > 0:
                if 0 < b[i] < self.buckets.size-1:
                    left = self.splines[b[i]-1](x, dx=dx)
                    right = self.splines[b[i]+1](x, dx=dx)
                    result[i] = 0.5*(right-v)/(self.buckets[b[i]+1]-self.buckets[b[i]]) + 0.5*(v-left)/(self.buckets[b[i]]-self.buckets[b[i]-1])
                elif b[i] > 0:
                    left = self.splines[b[i] - 1](x, dx=dx)
                    result[i] = (v - left) / (self.buckets[b[i]] - self.buckets[b[i] - 1])
                else:
                    right = self.splines[b[i] + 1](x, dx=dx)
                    result[i] = (right-v)/(self.buckets[b[i]+1]-self.buckets[b[i]])
            else:
                result[i] = v

        return result


