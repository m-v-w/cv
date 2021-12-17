import mckean
import numpy as np
from csaps import csaps

import mckeangenerator

h = 0.02
L = int(1 / h)
N = 1000
M = 100

# Matlab result: mean=0.4084 std=0.0087
#generator = mckeangenerator.SimpleGenerator()

result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
for j in range(M):
    Xr, _ = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    X, dW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    w = np.zeros((N, L+1))
    w[:, L] = np.mean(mckean.f_call(Xr[:, -1], Xr[:, -1]), axis=1)
    delta_phi = np.zeros((N, L))
    for l in range(L-1, 0, -1):
        order = np.argsort(Xr[:, l])
        spline = csaps(Xr[order, l], w[order, l+1], smooth=0.9)
        w[:, l] = spline(Xr[:, l])
        delta_phi[:, l] = spline(X[:, l], 1)
        dbg = np.mean((w[:, l] - w[:, l + 1]) ** 2)
        coeffs = spline.spline.coeffs
        print(
            '{l:d} dbg={dbg:.6f}, max-coeff={mc:.2f}, dmin={dmin:.2f}, dmax={dmax:.2f}'.format(
                l=l, dbg=dbg, mc=np.max(np.abs(coeffs)),
                dmin=np.min(delta_phi[:, l]), dmax=np.max(delta_phi[:, l])))

    cv = np.zeros((N, 1))
    for l in range(L):
        F_est = -np.mean(delta_phi[:, l]*mckean.b(X[:, l], X[:, l]), axis=1)
        cv = cv + F_est*dW[:, l+1]
    result_mc[j] = np.mean(mckean.f_call(X[:, -1], X[:, -1]))
    result_mc_cv[j] = np.mean(np.mean(mckean.f_call(X[:, -1], X[:, -1]), axis=1) + cv)
    print('smc={smc:.4f}, cv={vcv:.4f}, cv_mean={cvmean:.6f}'.format(smc=result_mc[j], vcv=result_mc_cv[j], cvmean=np.mean(cv)))

np.savez("data/spline_fcall.npz", result_mc, result_mc_cv)
print(np.mean(result_mc))
print(np.std(result_mc))
print(np.mean(result_mc_cv))
print(np.std(result_mc_cv))
