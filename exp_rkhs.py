import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import lsv_rkhs
import marketvol
import lsv
import py_vollib.black_scholes_merton.implied_volatility as impl

h = 1 / 365
L = int(5 / h)
N = 10000
M = 100
r = 0
kappa = 9
v0 = 0.16*0.16
theta = 0.16*0.16
xi = 0.4
rho = -0.5

lsv_rkhs = lsv_rkhs.LsvGenerator(r, kappa, v0, theta, xi, rho)
lsv_spline = lsv.LsvGenerator(r, kappa, v0, theta, xi, rho)
X_rkhs, _ = lsv_rkhs.generate(N, L, h)
X_spline, _ = lsv_spline.generate(N, L, h)
x_rkhs = X_rkhs[:, :, 0]
x_spline = X_spline[:, :, 0]

strikes = np.exp(lsv_rkhs.market_vol.moneyness)*lsv_rkhs.market_vol.s0
strike_grid, time_idx_grid = np.meshgrid(strikes[2:-1], range(30, L+1, 10))
simulated_prices_rkhs = np.mean(np.fmax(x_rkhs[:, time_idx_grid]-np.tile(strike_grid, (N, 1, 1)), 0), axis=0)
simulated_prices_spline = np.mean(np.fmax(x_spline[:, time_idx_grid]-np.tile(strike_grid, (N, 1, 1)), 0), axis=0)
simulated_itm_rkhs = np.sum(x_rkhs[:, time_idx_grid] > np.tile(strike_grid, (N, 1, 1)), axis=0)
simulated_itm_spline = np.sum(x_spline[:, time_idx_grid] > np.tile(strike_grid, (N, 1, 1)), axis=0)
simulated_iv_rkhs = np.zeros(strike_grid.shape)
simulated_iv_spline = np.zeros(strike_grid.shape)
market_iv = np.zeros(strike_grid.shape)
local_vol = np.zeros(strike_grid.shape)
for i in range(strike_grid.shape[0]):
    for j in range(strike_grid.shape[1]):
        y = np.log(strike_grid[i, j] / lsv_rkhs.market_vol.s0)
        t = time_idx_grid[i, j]*h
        market_iv[i, j] = np.sqrt(lsv_rkhs.market_vol.eval(y, t) / t)
        local_vol[i, j] = lsv_rkhs.market_vol.dupire_vol(t, strike_grid[i, j])
        if simulated_itm_rkhs[i, j] > 10 and simulated_prices_rkhs[i, j] > lsv_rkhs.market_vol.s0 - strike_grid[i, j]:
            simulated_iv_rkhs[i, j] = lsv_rkhs.market_vol.implied_volatility(simulated_prices_rkhs[i, j], lsv_rkhs.market_vol.s0, strike_grid[i, j], time_idx_grid[i, j]*h, 0, 0, 'c')
        else:
            simulated_iv_rkhs[i, j] = np.nan
        if simulated_itm_spline[i, j] > 10 and simulated_prices_spline[i, j] > lsv_rkhs.market_vol.s0 - strike_grid[i, j]:
            simulated_iv_spline[i, j] = lsv_rkhs.market_vol.implied_volatility(simulated_prices_spline[i, j], lsv_rkhs.market_vol.s0, strike_grid[i, j], time_idx_grid[i, j] * h, 0, 0,'c')
        else:
            simulated_iv_spline[i, j] = np.nan


idx = int((365-30)/10)
plt.plot(strikes[2:-1], simulated_iv_rkhs[idx, :], label='RKHS')
plt.plot(strikes[2:-1], simulated_iv_spline[idx, :], label='Spline')
plt.plot(strikes[2:-1], market_iv[idx, :], label='Market')
plt.legend()
plt.savefig("plots/lsv_1y_rkhs_vs_spline.png", format="png")
plt.savefig("plots/lsv_1y_rkhs_vs_spline.eps", format="eps")
plt.show()

