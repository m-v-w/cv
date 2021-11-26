import numpy as np
import pandas as pd
import scipy.interpolate
import py_vollib.black_scholes_merton.implied_volatility as impl


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

    @staticmethod
    def load_csv():
        df = pd.read_csv("data/spx_24112021.csv", names=['expiry', 'strike', 'time', 'last', 'bid', 'ask'])
        times = np.array((pd.to_datetime(df["expiry"]) - pd.to_datetime("2021-11-24 16:00:00")).astype('timedelta64[h]') / (24 * 365))
        strikes = np.array(df["strike"])
        prices = np.array(df["last"])
        s0 = 4671.2
        return MarketVol(times, strikes, prices, s0)

    def __init__(self, times, strikes, prices, s0):
        time_axis, time_map, time_counts = np.unique(times, return_inverse=True, return_counts=True)
        strike_axis, strike_map, strike_counts = np.unique(strikes, return_inverse=True, return_counts=True)
        implied_black_var = np.zeros((time_axis.size, strike_axis.size))
        for i in range(prices.size):
            time_idx = time_map[i]
            strike_idx = strike_map[i]
            if prices[i] > s0-strike_axis[strike_idx]+0.01:
                v = impl.implied_volatility(prices[i], s0, strike_axis[strike_idx], time_axis[time_idx], 0, 0, 'c')
                implied_black_var[time_idx, strike_idx] = v*v*times[i]

        min_strikes = time_axis.size / 2
        min_times = strike_axis.size / 4
        strike_filter = np.logical_or(strike_counts < min_strikes, np.abs(np.log(strike_axis / s0)) > 0.7)
        implied_black_var = np.delete(implied_black_var, strike_filter, axis=1)
        strike_axis = np.delete(strike_axis, strike_filter)
        implied_black_var = np.delete(implied_black_var, time_counts < min_times, axis=0)
        time_axis = np.delete(time_axis, time_counts < min_times)
        for j in range(strike_axis.size):
            if implied_black_var[0, j] <= np.min(implied_black_var[:, j]):
                implied_black_var[0, j] = np.min(implied_black_var[:, j])
            for i in range(1, time_axis.size):
                if implied_black_var[i, j] < implied_black_var[i-1, j]:
                    s = implied_black_var[(i-1):, j]
                    implied_black_var[i, j] = max(np.min(s), implied_black_var[i-1, j])

        self.spline = scipy.interpolate.interp2d(strike_axis, time_axis, implied_black_var, kind='cubic', copy=True,
                                            bounds_error=False, fill_value=None)

    def fix_slope(self):
        '''
     public void fixSurface(double[] vector, Date evalDate,Date[] expirations, DayCounter dayCounter,double strike) throws JOptimizerException {
        double[] t = new double[expirations.length];
        double[][] P = new double[vector.length][vector.length];
        double[] y = new double[vector.length];
        ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[y.length];
        double[] g = new double[y.length];
        double[] x0 = new double[y.length];
        g[0] = -1;
        // x_j <= x_{j+1}
        inequalities[0] = new LinearMultivariateRealFunction(g, 0);
        t[0] = dayCounter.yearFraction(evalDate,expirations[0]);
        boolean inconsistent = false;
        for (int j = 0; j < vector.length; j++) {
            if(j < y.length-1)
                t[j+1] = dayCounter.yearFraction(evalDate,expirations[j+1]);
            P[j][j]=1;
            y[j] = -vector[j] * vector[j] * t[j];
            x0[j] = -y[j];
            if(j > 0 && y[j] > y[j-1]) {
                logger.info("order on strike " + strike + " " + expirations[j - 1] + " (" + t[j - 1] + " years) " + (-y[j - 1]) + " vs. "
                        + expirations[j] + " (" + t[j] + " years)" + (-y[j]));
                inconsistent=true;
            }
            if(j < y.length-1) {
                g = new double[y.length];
                g[j] = 1;
                g[j + 1] = -1;
                // x_j <= x_{j+1}
                inequalities[j+1] = new LinearMultivariateRealFunction(g, 0);
            }
        }
        if(inconsistent) {
            PDQuadraticMultivariateRealFunction objectiveFunction = new PDQuadraticMultivariateRealFunction(P, y, 0);

            OptimizationRequest or = new OptimizationRequest();
            or.setF0(objectiveFunction);
            or.setToleranceFeas(1.E-12);
            or.setTolerance(1.E-12);
            or.setFi(inequalities);
            or.setInitialPoint(x0);

            //optimization
            JOptimizer opt = new JOptimizer();
            opt.setOptimizationRequest(or);
            double[] x;
            try {
                opt.optimize();
                x = opt.getOptimizationResponse().getSolution();
            } catch(JOptimizerException e) {
                x = fixSurfaceFailover(t,x0);
            }

            for (int j = 0; j < vector.length; j++)
                vector[j] = Math.sqrt(x[j] / t[j]);
        }
    }'''
        
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

