import numpy as np
import lsv
import csaps
import scipy.interpolate
import mckeangenerator
import splines.BucketSpline
import splines.GridSpline
import splines.SmoothSpline
from payouts import CallPayout
from runner import SimulationArgs


def spline_run(args: SimulationArgs, return_derivative=False):
    N = args.N
    Nr = args.Nr
    L = args.L
    generator = args.generator
    h = args.h
    payout = args.payout
    d_x, d_w = generator.get_dimensions()
    Xr, _ = generator.generate(Nr, L, h)
    X, dW = generator.generate(N, L, h)
    w = np.zeros((Nr, L+1))
    w[:, L] = payout(Xr)
    delta_phi = np.zeros((N, L + 1, d_x))
    delta_phi[:, -1, :] = payout.gradient(X)
    coeff = np.zeros((L, 8*d_x))
    for l in range(L-1, 0, -1):
        #spline = splines.BucketSpline.BucketSpline(Xr[:, l, :], w[:, l + 1], y_buckets=y_buckets, smooth=smooth, normalizedsmooth=False)
        if d_x == 2:
            spline = splines.GridSpline.GridSpline(Xr[:, l, :], w[:, l + 1])
        elif d_x == 1:
            spline = splines.SmoothSpline.SmoothSpline(Xr[:, l, 0], w[:, l + 1])
        else:
            raise ValueError("unsupported dimensions")
        w[:, l] = spline(Xr[:, l, :])
        dbg = np.mean((w[:, l]-w[:, l+1])**2)
        for k in range(d_x):
            delta_phi[:, l, k] = spline(X[:, l, :], daxis=k)
        c = spline.get_coeffs()
        coeff[l, :] = c.flatten()
        print('{l:d} dbg={dbg:.6f}, res={res:.2f}, max-coeff={mc:.2f}, dmin={dmin:.2f}, dmax={dmax:.2f}'.format(l=l, dbg=dbg, res=spline.get_residual(), mc=np.max(np.abs(spline.get_coeffs())), dmin=np.min(delta_phi[:, l, 0]), dmax=np.max(delta_phi[:, l, 0])))

    cv = np.zeros(N)
    for l in range(L):
        diffusion = generator.diffusion(X[:, l, :], l * h)  # TODO
        for k1 in range(d_x):
            for k2 in range(d_w):
                ff = np.mean(delta_phi[:, l, k1])
                F_est = -ff * diffusion[:, k1, k2]
                cv = cv + F_est * dW[:, l, k2]  # TODO dW[l+1] ???
    f_T = payout(X)
    result_mc = np.mean(f_T)
    result_mc_cv = np.mean(f_T)+np.sum(cv)
    result_mc_cv_mean = np.mean(cv)
    print('smc={smc:.4f}, cv={vcv:.4f}, cv_mean={cvmean:.6f}'.format(smc=result_mc, vcv=result_mc_cv,
                                                                     cvmean=result_mc_cv_mean))
    if return_derivative:
        return result_mc, result_mc_cv, result_mc_cv_mean, delta_phi, coeff
    else:
        return result_mc, result_mc_cv, result_mc_cv_mean
