import numpy as np
import lsv
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs


def flat_smc_run(args: SimulationArgs):
    N = args.N
    L = args.L
    generator = args.generator
    h = args.h
    payout = args.payout
    d_x, d_w = generator.get_dimensions()
    x_flat, _ = generator.generate(N, L, h)
    payout_flat = payout(x_flat)
    flat = np.mean(payout_flat)
    print('smc={smc:.4f}'.format(smc=flat))
    return flat, flat, 0

