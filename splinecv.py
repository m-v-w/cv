import numpy as np
import lsv
import csaps
import scipy.interpolate

import mckeangenerator
import splines.BucketSpline
import splines.GridSpline
from payouts import CallPayout

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

generator = lsv.LsvGenerator(r, kappa, v0, theta, xi, rho)
strike = generator.market_vol.s0
#generator = mckeangenerator.SimpleCorrGenerator(-0.5)
#strike = 0.5
payout = CallPayout(strike)
h = 0.02
L = int(1 / h)
N = 1000
M = 100

maturity_idx = -1
#smooth = 0.0000000001
smooth = 0.5
y_buckets = 5
result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_mc_cv_mean = np.zeros(M)
d_x, d_w = generator.get_dimensions()
for j in range(M):
    Xr, _ = generator.generate(N, L, h)
    X, dW = generator.generate(N, L, h)
    w = np.zeros((N, L+1))
    w[:, L] = payout(Xr)
    delta_phi = np.zeros((N, L, d_x))
    for l in range(L-1, 0, -1):
        s = N * np.std(w[:, l + 1])
        #spline = splines.BucketSpline.BucketSpline(Xr[:, l, :], w[:, l + 1], y_buckets=y_buckets, smooth=smooth, normalizedsmooth=False)
        spline = splines.GridSpline.GridSpline(Xr[:, l, :], w[:, l + 1])
        w[:, l] = spline(Xr[:, l, :])
        dbg = np.mean((w[:, l]-w[:, l+1])**2)
        for k in range(d_x):
            delta_phi[:, l, k] = spline(X[:, l, :], daxis=k)
        print('{l:d} s={s:.2f}, dbg={dbg:.6f}, res={res:.2f}, max-coeff={mc:.2f}, dmin={dmin:.2f}, dmax={dmax:.2f}'.format(l=l, s=s, dbg=dbg, res=spline.get_residual(), mc=np.max(np.abs(spline.get_coeffs())), dmin=np.min(delta_phi[:, l, 0]), dmax=np.max(delta_phi[:, l, 0])))

    cv = np.zeros(N)
    for l in range(L):
        diffusion = generator.diffusion(X[:, l, :], l * h)
        for k1 in range(d_x):
            for k2 in range(d_w):
                F_est = -delta_phi[:, l, k1]*diffusion[:, k1, k2]
                cv = cv + F_est * dW[:, l + 1, k2]
    f_T = payout(X)
    result_mc[j] = np.mean(f_T)
    result_mc_cv[j] = np.mean(f_T + cv)
    result_mc_cv_mean[j] = np.mean(cv)
    print('{j:d} smc={smc:.4f}, cv={vcv:.4f}, cv_mean={cvmean:.6f}'.format(j=j, smc=result_mc[j], vcv=result_mc_cv[j],
                                                                     cvmean=result_mc_cv_mean[j]))

np.savez("data/spline_lsv.npz", result_mc, result_mc_cv)
print(smooth)
print(np.mean(result_mc))
print(np.std(result_mc))
print(np.mean(result_mc_cv))
print(np.std(result_mc_cv))
print(np.mean(result_mc_cv_mean))
print(np.std(result_mc_cv_mean))