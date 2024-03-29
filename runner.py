import mckeangenerator
from payouts import CallPayout
import numpy as np


class SimulationArgs(object):

    def __init__(self, generator=mckeangenerator.SimpleGenerator(), payout=CallPayout(0.5)):
        self.T = 1
        self.L = 50
        self.h = self.T / self.L
        self.N = 4992
        self.Nr = 4992
        self.K = 5
        self.generator = generator
        payout.h = self.h
        self.payout = payout



def print_results(simulation_name, result_mc, result_mc_cv, result_mc_cv_mean, args: SimulationArgs, report_file):
    M = result_mc.shape[0]
    title = "{simulation_name:s} {gen_name:s} {payout:s} N={n:d} N_R={n_r:d} M={m:d} L={l:d}".format(simulation_name=simulation_name, gen_name=args.generator.name, payout=args.payout.name, n=args.N, n_r=args.Nr, m=M, l=args.L)
    print(title, file=report_file)
    print('MC Ref: mean=%2.6f std=%2.6f' % (np.mean(result_mc), np.std(result_mc)), file=report_file)
    print('MC CV: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv), np.std(result_mc_cv)), file=report_file)
    print('MC CV-Mean: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv_mean), np.std(result_mc_cv_mean)), file=report_file)
    print('Gain: %.6f' % (np.std(result_mc) / np.std(result_mc_cv)),
          file=report_file)
    print("", file=report_file)

