import numpy as np
import lsv
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout

h = 1 / 100
L = int(5 / h)
N = 1000
M = 100

#generator = lsv.LsvGenerator()
#strike = generator.market_vol.s0
generator = mckeangenerator.SimpleCorrGenerator(-0.5)
strike = 0.5
#payout = CallPayout(strike)
payout = TrigonometricPayout(h, generator)

result_mc = np.zeros(M)
d_x, d_w = generator.get_dimensions()
n_split = np.power(N, 1 / 3)
n_outer = round(n_split)
n_inner = round(n_split ** 2)
for j in range(M):
    outer_result = np.empty(n_outer)
    for i in range(n_outer):
        X, _ = generator.generate(n_inner, L, h)
        f_T = payout(X)
        outer_result[i] = np.mean(f_T)
    result_mc[j] = np.mean(outer_result)

np.savez("data/smc_"+generator.name+"_"+payout.name+".npz", result_mc)
print(np.mean(result_mc))
print(np.std(result_mc))
