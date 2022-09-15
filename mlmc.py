import numpy as np
import lsv
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs


def mlmc_run(args: SimulationArgs):
    payout = args.payout
    budget = args.N*args.N*args.L
    beta_L = 2
    beta_N = 1
    beta_M = 1
    factor_N = args.N
    factor_M = 1
    b = beta_M*beta_L*beta_N*beta_N
    c = budget/(factor_M*factor_N**2)
    if b == 1:
        total_levels = round((np.sqrt(1+8*c)-1)/2)
    else:
        total_levels = round(np.log(1-c*(1-b))/np.log(b)-1)
    generator = args.generator

    d_x, d_w = generator.get_dimensions()
    levels = np.zeros(total_levels)
    for level in range(1, total_levels+1):
        M_level = max(1, round(factor_M * np.power(beta_M, level)))
        L_fine = beta_L ** level
        N_fine = int(factor_N * beta_N ** level)
        h_fine = args.T / L_fine
        L_coarse = beta_L ** (level - 1)
        N_coarse = int(factor_N * beta_N ** (level - 1))
        h_coarse = args.T / L_coarse
        samples = np.empty(M_level)
        for m in range(M_level):
            dW_fine = np.math.sqrt(h_fine) * np.random.normal(0, 1, (N_fine, L_fine, 1))
            dW_coarse = (dW_fine[:N_coarse, :-1, :] + dW_fine[:N_coarse, 1:, :])[:, ::2, :]
            x_fine = generator.calc_values(dW_fine, h_fine)
            x_coarse = generator.calc_values(dW_coarse, h_coarse)
            if level > 1:
                samples[m] = np.mean(payout(x_fine)) - np.mean(payout(x_coarse))
            else:
                samples[m] = np.mean(payout(x_fine))
        levels[level - 1] = np.mean(samples)

    result = np.sum(levels)
    print('result={result:.4f} levels={levels:d}'.format(result=result, levels=total_levels))
    return result, result, 0
# return flat, v, 0

# mlmc_run(SimulationArgs())
