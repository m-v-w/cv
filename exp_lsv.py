import numpy as np
import lsv
import csaps
import scipy.interpolate
import splines.BucketSpline

h = 1 / 100
L = int(5 / h)
N = 1000
M = 100
r = 0
kappa = 9
v0 = 0.16*0.16
theta = 0.16*0.16
xi = 0.4
rho = -0.5

lsv = lsv.LsvGenerator(r, kappa, v0, theta, xi, rho)
strike = lsv.market_vol.s0
maturity_idx = -1

result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
for j in range(M):
    Xr, _ = lsv.generate(N, L, h)
    X, dW = lsv.generate(N, L, h)
    w = np.zeros((N, L+1))
    w[:, L] = np.fmax(Xr[:, maturity_idx, 0]-strike, 0)
    delta_phi = np.zeros((N, L, 2))
    space_x = np.linspace(np.min(Xr[:, :, 0]), np.max(Xr[:, :, 0]), num=50)
    space_y = np.linspace(np.min(Xr[:, :, 1]), np.max(Xr[:, :, 1]), num=50)
    for l in range(L-1, 0, -1):
        #s = N * np.std(w[:, l + 1])
        #spline = scipy.interpolate.SmoothBivariateSpline(Xr[:, l, 0], Xr[:, l, 1], w[:, l + 1], s=s)
        #input = [Xr[:, l, 0], Xr[:, l, 1]]
        #spline = csaps.NdGridCubicSmoothingSpline(input, w[:, l + 1], smooth=.9)
        spline = splines.BucketSpline.BucketSpline(Xr[:, l, 0], Xr[:, l, 1], w[:, l + 1])
        w[:, l] = spline(Xr[:, l, 0], Xr[:, l, 1], grid=False)
        dbg = np.mean((w[:,l]-w[:,l+1])**2)
        delta_phi[:, l, 0] = spline(Xr[:, l, 0], Xr[:, l, 1], dx=1)
        delta_phi[:, l, 1] = spline(Xr[:, l, 0], Xr[:, l, 1], dy=1)
        print('{l:d} s={s:.2f}, dbg={dbg:.2f}, res={res:.2f}, max-coeff={mc:.2f}, dmin={dmin:.2f}, dmax={dmax:.2f}'.format(l=l, s=s, dbg=dbg, res=spline.get_residual(), mc=np.max(np.abs(spline.get_coeffs())), dmin=np.min(delta_phi[:, l, 0]), dmax=np.max(delta_phi[:, l, 0])))

    cv = np.zeros((N, 1))
    for l in range(L):
        F_est = -delta_phi[:, l]*lsv.diffusion(X[:, l, :], l*h)
        cv = cv + F_est*dW[:, l+1]
    result_mc[j] = np.mean(np.fmax(X[:, maturity_idx]-strike, 0))
    result_mc_cv[j] = np.mean(np.fmax(X[:, maturity_idx]-strike, 0) + cv)
    print(result_mc_cv[j])

np.savez("data/spline_lsv.npz", result_mc, result_mc_cv)
print(np.mean(result_mc))
print(np.std(result_mc))
print(np.mean(result_mc_cv))
print(np.std(result_mc_cv))