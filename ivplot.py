import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import marketvol

market_vol = marketvol.MarketVol.load_csv()
local_vol, strike_space, time_space = market_vol.create_mesh()
X, Y = np.meshgrid(strike_space, time_space)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, local_vol.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()


#x_space = np.linspace(market_vol.spline.x_min, market_vol.spline.x_max, 100)
#y_space = np.linspace(market_vol.spline.y_min, market_vol.spline.y_max, 100)
local_implied = np.zeros(X.shape)
var_matrix = np.zeros(X.shape)
for i in range(local_implied.shape[0]):
    for j in range(local_implied.shape[1]):
        var_matrix[i, j] = market_vol.eval(X[i, j], Y[i, j])
        if Y[i, j] > 0:
            local_implied[i, j] = np.sqrt(var_matrix[i, j])/Y[i, j]


X, Y = np.meshgrid(strike_space, time_space)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, local_implied, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

