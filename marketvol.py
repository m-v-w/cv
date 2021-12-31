import numpy as np
import pandas as pd
import scipy.interpolate
import py_vollib.black_scholes_merton.implied_volatility as impl
import csaps


class MarketVol(object):
    @staticmethod
    def load_npz(file):
        data = np.load(file)
        data_prices = data['arr_0']
        data_strikes = data['arr_1']
        data_dates = data['arr_2']
        s0 = data['arr_3']
        t0 = data['arr_4']
        s0 = min(s0, data_strikes[0] + data_prices[0, 0] - 1)
        data_times = (data_dates - t0).astype(int) / 365
        return MarketVol()

    @staticmethod
    def load_csv():
        df = pd.read_csv("data/spx_24112021.csv", names=['expiry', 'strike', 'time', 'last', 'bid', 'ask'])
        times = np.array((pd.to_datetime(df["expiry"]) - pd.to_datetime("2021-11-24 16:00:00")).astype('timedelta64[h]') / (24 * 365))
        strikes = np.array(df["strike"])
        prices = np.array(df["ask"])
        s0 = 4671.2
        return MarketVol(times, strikes, prices, s0)

    def init_matrix(self, times, strikes, prices, s0):
        implied_vol = np.zeros((times.size, strikes.size))
        for i in range(times.size):
            for j in range(strikes.size):
                if prices[i, j] > s0 - strikes[j] + 0.01:
                    v = impl.implied_volatility(prices[i, j], s0, strikes[j], times[i], 0, 0, 'c')
                    implied_vol[i, j] = v

    def __init__(self, times, strikes, prices, s0):
        time_axis, time_map, time_counts = np.unique(times, return_inverse=True, return_counts=True)
        strike_axis, strike_map, strike_counts = np.unique(strikes, return_inverse=True, return_counts=True)
        implied_vol = np.zeros((time_axis.size, strike_axis.size))
        for i in range(prices.size):
            time_idx = time_map[i]
            strike_idx = strike_map[i]
            if prices[i] > s0-strike_axis[strike_idx]+0.01:
                v = impl.implied_volatility(prices[i], s0, strike_axis[strike_idx], time_axis[time_idx], 0, 0, 'c')
                implied_vol[time_idx, strike_idx] = v

        # clearup esoteric strikes and maturities
        min_strikes = time_axis.size / 2
        strike_filter = np.logical_or(strike_counts < min_strikes, np.abs(np.log(strike_axis / s0)) > 0.7)
        implied_vol = np.delete(implied_vol, strike_filter, axis=1)
        strike_axis = np.delete(strike_axis, strike_filter)
        min_times = strike_axis.size / 3
        time_filter = np.logical_or(time_counts < min_times, time_axis < 0.1)
        implied_vol = np.delete(implied_vol, time_filter, axis=0)
        time_axis = np.delete(time_axis, time_filter)

        # ensure variance increases over time
        forward = s0
        moneyness = np.log(strike_axis/forward)
        var_matrix = np.zeros((time_axis.size, strike_axis.size))
        min_dt = 0.01*0.01
        for j in range(moneyness.size):
            if implied_vol[0, j] == 0:
                s = implied_vol[0:, j]
                implied_vol[0, j] = np.min(s[s > 0])
            var_matrix[0, j] = implied_vol[0, j]**2*time_axis[0]
            for i in range(1, time_axis.size):
                if implied_vol[i, j] == 0 and 0 < j < moneyness.size - 1:
                    left_array = implied_vol[i, :j]
                    right_array = implied_vol[i, j+1:]
                    if np.sum(left_array) > 0 and np.sum(right_array) > 0:
                        left_idx = np.nonzero(left_array)[0][-1]
                        right_idx = np.nonzero(right_array)[0][0]+j+1
                        t = (moneyness[j]-moneyness[left_idx])/(moneyness[right_idx]-moneyness[left_idx])
                        implied_vol[i, j] = t*implied_vol[i, right_idx] + (1-t)*implied_vol[i, left_idx]
                var_lower_bound = implied_vol[i-1, j]**2*time_axis[i-1]+(time_axis[i]-time_axis[i-1])*min_dt
                if implied_vol[i, j]**2*time_axis[i] < var_lower_bound:
                    var = var_lower_bound
                    future_vars = implied_vol[i:, j]**2*time_axis[i:]
                    if np.sum(future_vars) > 0:
                        future_var_min = np.min(future_vars[future_vars > 0])
                        if future_var_min > var_lower_bound:
                            var = 0.5*(future_var_min + var_lower_bound)
                    implied_vol[i, j] = np.sqrt(var / time_axis[i])
                    var_matrix[i, j] = var
                else:
                    var_matrix[i, j] = implied_vol[i, j]**2*time_axis[i]
        self.var_matrix = np.zeros((var_matrix.shape[0]+1, var_matrix.shape[1]))
        self.var_matrix[1:, :] = var_matrix
        self.vol_matrix = implied_vol
        self.moneyness = moneyness
        self.times = np.zeros(time_axis.size+1)
        self.times[1:] = time_axis
        x_grid, y_grid = np.meshgrid(self.moneyness, self.times)
        x = x_grid.flatten()
        y = y_grid.flatten()
        z = self.var_matrix.flatten()
        #self.spline = csaps.NdGridCubicSmoothingSpline([self.times, self.moneyness], self.var_matrix, smooth=.9)
        self.spline = scipy.interpolate.SmoothBivariateSpline(x, y, z, s=z.size*np.std(z))
        # self.spline = scipy.interpolate.interp2d(moneyness, self.times, self.var_matrix, kind='cubic', copy=True,
        #                                    bounds_error=False, fill_value=None)
        self.s0 = s0

    def eval(self, y, t, dx=0, dy=0):
        if t < self.times[1]:
            v = self.spline(y, self.times[1], dx, dy)
            if dy == 0:
                return v
            elif dy == 1:
                return v/(self.times[1]-self.times[0])
            else:
                return 0
        if y < self.moneyness[1]:
            v = self.spline(self.moneyness[1], t, dx, dy)
            if dx == 0:
                return v
            else:
                return 0
        #return self.spline(y, t, (dx, dy))
        return self.spline(y, t, dx, dy)

    def eval_linear(self, y, t, dx=0, dy=0):
        if dx != 0 and dy != 0:
            raise ValueError("not supported")
        if dx >= 2 or dy >= 2:
            return 0
        t_upper = np.nonzero(self.times >= t)[0][0]
        t_lower = np.nonzero(self.times <= t)[0][-1]
        y_upper = np.nonzero(self.moneyness >= y)[0][0]
        y_lower = np.nonzero(self.moneyness <= y)[0][-1]
        tv=yv=0
        if t_upper == t_lower and dy > 0:
            t_upper = min(self.times.size - 1, t_upper + 1)
            t_lower = max(0, t_lower - 1)
        if y_upper == y_lower and dx > 0:
            y_upper = min(self.moneyness.size - 1, y_upper + 1)
            y_lower = max(0, y_lower - 1)
        if t_upper > t_lower:
            tv = 1 - (t - self.times[t_lower]) / (self.times[t_upper] - self.times[t_lower])
        if y_upper > y_lower:
            yv = 1 - (y - self.moneyness[y_lower]) / (self.moneyness[y_upper] - self.moneyness[y_lower])
        yl = tv * self.var_matrix[t_lower, y_lower] + (1 - tv) * self.var_matrix[t_upper, y_lower]
        yu = tv * self.var_matrix[t_lower, y_upper] + (1 - tv) * self.var_matrix[t_upper, y_upper]
        tl = yv * self.var_matrix[t_lower, y_lower] + (1 - yv) * self.var_matrix[t_lower, y_upper]
        tu = yv * self.var_matrix[t_upper, y_lower] + (1 - yv) * self.var_matrix[t_upper, y_upper]
        if dx == 0 and dy == 0:
            a = yv*yl+(1-yv)*yu
            b = tv*tl+(1-tv)*tu
            return a
        if dx == 1: # =dy
            return (yu-yl)/(self.moneyness[y_upper] - self.moneyness[y_lower])
        if dy == 1:
            return (tu - tl) / (self.times[t_upper] - self.times[t_lower])

    def create_mesh(self):
        x_space = np.linspace(self.moneyness[1], self.moneyness[-1]*1.5, 100)
        y_space = np.linspace(self.times[1], self.times[-1], 100)
        local_vol = np.zeros((x_space.size, y_space.size))
        for i in range(x_space.size):
            x = np.exp(x_space[i]) * self.s0
            for j in range(y_space.size):
                local_vol[i, j] = self.dupire_vol(y_space[j], x)
        return local_vol, x_space, y_space

    def dupire_vol(self, t, x):
        forward = self.s0
        y = np.log(x/forward)
        t = min(self.times[-2], max(t, self.times[1]))
        y = min(self.moneyness[-2], max(y, self.moneyness[1]))
        dwdt = self.eval(y, t, dx=0, dy=1)
        if (dwdt <= 0):
            raise ValueError("local vol not calculated, variance is decreasing")
        dwdy = self.eval(y, t, dx=1, dy=0)
        d2wdy2 = self.eval(y, t, dx=2, dy=0)
        w = self.eval(y, t, dx=0, dy=0)
        if dwdy == 0.0 and d2wdy2 == 0:
            return np.sqrt(dwdt)
        if w <= 0:
            return np.sqrt(dwdt)
        den1 = 1.0 - y / w * dwdy
        den2 = 0.25 * (-0.25 - 1.0 / w + y * y / w / w) * dwdy * dwdy
        den3 = 0.5 * d2wdy2
        den = den1 + den2 + den3
        if den <= 0:
            return np.sqrt(dwdt)
            # raise ValueError("negative local vol^2 at strike, the black vol surface is not smooth enough")
        result = dwdt / den
        return np.sqrt(result)

