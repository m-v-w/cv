import numpy as np
import lsv
import mckeangenerator
import multiprocessing
from payouts import CallPayout
from runner import SimulationArgs
from splines.polygons import hermite_poly, hermite_deriv_poly, hermite_poly_multi, hermite_poly_eval_deriv


def poly_run(args: SimulationArgs, return_derivative=False):
    N = args.N
    #Nr = args.Nr
    Nr = N
    L = args.L
    generator = args.generator
    h = args.h
    payout = args.payout
    K = args.K
    d_x, d_w = generator.get_dimensions()
    #Xr, _ = generator.generate(Nr, L, h)
    X, dW = generator.generate(N, L, h)
    Xr = X
    w = np.zeros((Nr, L+1))
    w[:, L] = payout(Xr)
    delta_phi = np.zeros((N, L + 1, d_x))
    delta_phi[:, -1, :] = payout.gradient(X)
    coeff = np.zeros((L, d_x*(K-1)+1))
    for l in range(L-1, 0, -1):
        base = hermite_poly_multi(Xr[:, l, :], K)
        fit, _, _, _ = np.linalg.lstsq(base, w[:, l + 1], rcond=None)
        coeff[l, :] = fit
        w[:, l] = base @ fit
        delta_phi[:, l, :] = hermite_poly_eval_deriv(X[:, l, :], K, fit)
        # print('{l:d} s={s:.2f}, dbg={dbg:.6f}, res={res:.2f}, max-coeff={mc:.2f}, dmin={dmin:.2f}, dmax={dmax:.2f}'.format(l=l, s=s, dbg=dbg, res=spline.get_residual(), mc=np.max(np.abs(spline.get_coeffs())), dmin=np.min(delta_phi[:, l, 0]), dmax=np.max(delta_phi[:, l, 0])))

    cv = np.zeros(N)
    for l in range(L):
        diffusion = generator.diffusion(X[:, l, :], l * h)
        for k1 in range(d_x):
            for k2 in range(d_w):
                F_est = -delta_phi[:, l, k1]*diffusion[:, k1, k2]
                cv = cv + F_est * dW[:, l + 1, k2]
    f_T = payout(X)
    if return_derivative:
        return np.mean(f_T), np.mean(f_T + cv), np.mean(cv), delta_phi, coeff
    else:
        return np.mean(f_T), np.mean(f_T + cv), np.mean(cv)
