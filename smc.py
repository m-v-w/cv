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
    n_split = np.power(N, 1 / 3)
    n_outer = round(n_split)
    n_inner = round(n_split ** 2)
    outer_result = np.empty(n_outer)
    for i in range(n_outer):
        X, _ = generator.generate(n_inner, L, h)
        f_T = payout(X)
        outer_result[i] = np.mean(f_T)
    v = np.mean(outer_result)
    print('smc={smc:.4f}'.format(smc=v))
    return v, v, 0


