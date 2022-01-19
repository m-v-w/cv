import sys

import lsv_rkhs
import mckeangenerator
# from nncv_eval import nncv_run
from payouts import CallPayout, TrigonometricPayout, SquaredPayout
import lsv
from ploycv import poly_run
import numpy as np
import multiprocessing
from runner import SimulationArgs, print_results
from smc import smc_run
from splinecv import spline_run

if __name__ == '__main__':
    simulation_name = "spline"
    M = 100
    #generator = mckeangenerator.SimpleGenerator()
    #generator = mckeangenerator.SimpleCorrGenerator(-0.5)
    #payout = CallPayout(0.5)
    #payout = SquaredPayout()
    generator = lsv_rkhs.LsvGenerator()
    payout = CallPayout(generator.market_vol.s0)

    args = SimulationArgs(generator, payout)
    arg_list = [args] * M
    pool_obj = multiprocessing.Pool(processes=6)
    result_list = pool_obj.map(spline_run, arg_list)

    result_mc = np.array([result_list[i][0] for i in range(M)])
    result_mc_cv = np.array([result_list[i][1] for i in range(M)])
    result_mc_cv_mean = np.array([result_list[i][2] for i in range(M)])
    np.savez("data/"+simulation_name+"_"+args.generator.name+"_"+args.payout.name+"_%d.npz" % args.L, result_mc, result_mc_cv, result_mc_cv_mean)
    print_results(simulation_name, result_mc, result_mc_cv, result_mc_cv_mean, args, sys.stdout)
    with open("data/report.txt", "a") as report_file:
        print_results(simulation_name, result_mc, result_mc_cv, result_mc_cv_mean, args, report_file)

