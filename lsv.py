import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import marketvol


market_vol = marketvol.MarketVol("data/spx.npz")
local_vol, x_space, y_space = market_vol.create_mesh()
x_space = np.linspace(market_vol.spline.x_min, market_vol.spline.x_max, 100)
y_space = np.linspace(market_vol.spline.y_min, market_vol.spline.y_max, 100)
local_implied = np.zeros((x_space.size, y_space.size))
for i in range(x_space.size):
    for j in range(y_space.size):
        local_implied[i, j] = market_vol.spline(x_space[i], y_space[j])

X, Y = np.meshgrid(x_space, y_space)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, local_implied, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()