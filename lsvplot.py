import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
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

lsv = lsv.LsvGenerator(r, kappa, v0, theta, xi, rho)
X, _ = lsv.generate(N, L, h)
x = X[:, :, 0]
v = X[:, :, 1]
x_local, _ = lsv.generate_localvol(N, L, h)

'''for i in range(5):
    plt.plot(x[i, :])
    plt.show()
plt.plot(np.sqrt(v[0, :]))
plt.show()'''

strikes = np.exp(lsv.market_vol.moneyness)*lsv.market_vol.s0
strike_grid, time_idx_grid = np.meshgrid(strikes[2:-1], range(30, L+1, 10))
simulated_prices = np.mean(np.fmax(x[:, time_idx_grid]-np.tile(strike_grid, (N, 1, 1)), 0), axis=0)
local_prices = np.mean(np.fmax(x_local[:, time_idx_grid]-np.tile(strike_grid, (N, 1, 1)), 0), axis=0)
simulated_itm = np.sum(x[:, time_idx_grid] > np.tile(strike_grid, (N, 1, 1)), axis=0)
local_itm = np.sum(x_local[:, time_idx_grid] > np.tile(strike_grid, (N, 1, 1)), axis=0)
simulated_iv = np.zeros(strike_grid.shape)
simulated_local_iv = np.zeros(strike_grid.shape)
market_iv = np.zeros(strike_grid.shape)
local_vol = np.zeros(strike_grid.shape)
for i in range(strike_grid.shape[0]):
    for j in range(strike_grid.shape[1]):
        y = np.log(strike_grid[i, j] / lsv.market_vol.s0)
        t = time_idx_grid[i, j]*h
        market_iv[i, j] = np.sqrt(lsv.market_vol.eval(y, t) / t)
        local_vol[i, j] = lsv.market_vol.dupire_vol(t, strike_grid[i, j])
        if simulated_itm[i, j] > 10 and simulated_prices[i, j] > lsv.market_vol.s0 - strike_grid[i, j]:
            simulated_iv[i, j] = impl.implied_volatility(simulated_prices[i, j], lsv.market_vol.s0, strike_grid[i, j], time_idx_grid[i, j]*h, 0, 0, 'c')
        else:
            simulated_iv[i, j] = np.nan
        if local_itm[i, j] > 10 and local_prices[i, j] > lsv.market_vol.s0 - strike_grid[i, j]:
            simulated_local_iv[i, j] = impl.implied_volatility(local_prices[i, j], lsv.market_vol.s0,
                                                         strike_grid[i, j], time_idx_grid[i, j] * h, 0, 0, 'c')
        else:
            simulated_local_iv[i, j] = np.nan



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.title("Simulated LSV, implied call vol")
surf = ax.plot_surface(strike_grid, time_idx_grid, simulated_iv, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("plots/sim_lsv_iv.png", format="png")
plt.savefig("plots/sim_lsv_iv.eps", format="eps")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.title("Market Black-Scholes implied call vol")
surf = ax.plot_surface(strike_grid, time_idx_grid, market_iv, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("plots/market_iv.png", format="png")
plt.savefig("plots/market_iv.eps", format="eps")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.title("Market Dupire local vol")
surf = ax.plot_surface(strike_grid, time_idx_grid, local_vol, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("plots/market_dupire.png", format="png")
plt.savefig("plots/market_dupire.eps", format="eps")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.title("Simulated local vol, implied call vol")
surf = ax.plot_surface(strike_grid, time_idx_grid, simulated_local_iv, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.savefig("plots/sim_lv_iv.png", format="png")
plt.savefig("plots/sim_lv_iv.eps", format="eps")
plt.show()
