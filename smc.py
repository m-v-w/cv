import numpy as np
import lsv
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs


def smc_run(args: SimulationArgs):
    N = args.N
    L = args.L
    generator = args.generator
    h = args.h
    payout = args.payout
    d_x, d_w = generator.get_dimensions()
    n_split = 4 # np.power(N, 1 / 10)
    n_outer = round(n_split)
    n_inner = round(N / n_outer)
    outer_result = np.empty(n_outer)
    for i in range(n_outer):
        X, _ = generator.generate(n_inner, L, h)
        f_T = payout(X)
        outer_result[i] = np.mean(f_T)
    v = np.mean(outer_result)
    x_flat, _ = generator.generate(N, L, h)
    payout_flat = payout(x_flat)
    flat = np.mean(payout_flat)
    print('smc={smc:.4f} flat={flat:.4f} outer={outer:d} inner={inner:d}'.format(smc=v, flat=flat, outer=n_outer, inner=n_inner))
    return flat, v, 0


