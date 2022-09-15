import numpy as np
import lsv
import mckeangenerator
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs

args = SimulationArgs()

L = args.L
generator = args.generator
h = args.h
payout = args.payout
d_x, d_w = generator.get_dimensions()
h_min = np.empty((L, 10))
for m in range(10):
    N = 100+m*100
    x, _ = generator.generate(N, L, h)
    for l in range(L):
        v = x[:, l+1, 0]
        v_max = np.max(v)
        mat = np.abs(v.reshape((-1, 1)) - v.reshape((1, -1)))+np.diag(v_max*np.ones(N))
        h_min[l, m] = np.min(mat)

