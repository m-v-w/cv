import numpy as np
import scipy.interpolate
import py_vollib.black_scholes_merton.implied_volatility as impl


class MarketVol(object):
    def __init__(self, file):
        data = np.load(file)
        data_prices = data['arr_0']
        data_strikes = data['arr_1']
        data_dates = data['arr_2']
        s0 = data['arr_3']
        t0 = data['arr_4']
        s0 = min(s0, data_strikes[0]+data_prices[0, 0]-1)
        data_times = (data_dates - t0).astype(int) / 365
        implied_black_var = np.zeros((data_times.size, data_strikes.size))
        for i in range(data_strikes.size):
            for j in range(data_times.size):
                #implied_volatility(price, S, K, t, r, q, flag)
                v = impl.implied_volatility(data_prices[j,i], s0, data_strikes[i], data_times[j], 0, 0, 'c')
                implied_black_var[j, i] = v*v*data_times[j]

        self.spline = scipy.interpolate.interp2d(data_strikes, data_times, implied_black_var, kind='cubic', copy=True,
                                            bounds_error=False, fill_value=None)

    def create_mesh(self):
        x_space = np.linspace(self.spline.x_min, self.spline.x_max, 100)
        y_space = np.linspace(self.spline.y_min, self.spline.y_max, 100)
        local_vol = np.zeros((x_space.size, y_space.size))
        for i in range(x_space.size):
            for j in range(y_space.size):
                local_vol[i, j] = self.dupire_vol(y_space[i], x_space[j])
        return local_vol, x_space, y_space

    def dupire_vol(self, t, x):
        dwdt = self.spline(x, t, dx=0, dy=1)
        if (dwdt < 0):
            if (dwdt < -0.01):
                raise ValueError("local vol not calculated, variance is decreasing")
            return 0
        dwdy = self.spline(x, t, dx=1, dy=0)
        d2wdy2 = self.spline(x, t, dx=2, dy=0)
        w = self.spline(x, t, dx=0, dy=0)
        if dwdy == 0.0 and d2wdy2 == 0:
            return np.sqrt(dwdt)
        den1 = 1.0 - x / w * dwdy
        den2 = 0.25 * (-0.25 - 1.0 / w + x * x / w / w) * dwdy * dwdy
        den3 = 0.5 * d2wdy2
        den = den1 + den2 + den3
        result = dwdt / den
        if (result < 0):
            raise ValueError("negative local vol^2 at strike); the black vol surface is not smooth enough")
        return np.sqrt(result)

