import math
import time

import numpy as np
import lsv
import lsv_rkhs
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs


def mlmc_dev_run(args: SimulationArgs):
    payout = args.payout

    budget = args.L*(args.N**2)
    totalLevels = 4
    #schedule_L = np.array([32,64,128,256,512])
    schedule_L = np.ones(totalLevels) * args.L
    t = math.ceil(math.log2(args.N))-totalLevels-1
    schedule_N = 2**np.array(range(t, t+totalLevels))
    p = 1
    schedule_M = 2 ** (-(p+4)*np.array(range(totalLevels))/2)
    scale_M = 0.5*budget/np.sum(schedule_M*schedule_L*args.generator.get_complexity(schedule_N))
    schedule_M = np.maximum(1, np.round(scale_M*schedule_M))
    used = np.sum(schedule_M*schedule_L*args.generator.get_complexity(schedule_N))
    print('budget={budget:.4f}, used={used:.4f}, rel={rel:.4f}'.format(budget=budget, used=used, rel=(used/budget)))
    #N = int(math.sqrt(budget/np.sum(schedule_L)))
    generator = args.generator
    L = args.L
    h = args.T / args.L
    d_x, d_w = generator.get_dimensions()
    levels = np.zeros(totalLevels)
    for level in range(totalLevels):
        M_level = int(schedule_M[level])
        N_fine = int(schedule_N[level])
        N_coarse = int(N_fine / 2)
        samples = np.empty(M_level)
        for m in range(M_level):
            dW_fine = np.math.sqrt(h) * np.random.normal(0, 1, (N_fine, L, d_w))
            dW_coarse_1 = dW_fine[:N_coarse, :, :]
            dW_coarse_2 = dW_fine[N_coarse:, :, :]
            x_fine = generator.calc_values(dW_fine, h)
            x_coarse_1 = generator.calc_values(dW_coarse_1, h)
            x_coarse_2 = generator.calc_values(dW_coarse_2, h)
            if level > 0:
                samples[m] = np.mean(payout(x_fine)) - 0.5 * (np.mean(payout(x_coarse_1)) + np.mean(payout(x_coarse_2)))
            else:
                samples[m] = np.mean(payout(x_fine))
        levels[level] = np.mean(samples)

    result = np.sum(levels)
    print('result={result:.4f}'.format(result=result))
    return result, result, 0


#start_time = time.time()
#generator = lsv_rkhs.LsvGenerator()
#payout = CallPayout(generator.market_vol.s0)
#mlmc_dev_run(SimulationArgs(generator, payout))
#print("--- %s seconds ---" % (time.time() - start_time))
