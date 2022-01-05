import sys
import mckeangenerator
from nncv_eval import nncv_run
from payouts import CallPayout
import lsv
from ploycv import poly_run
import numpy as np
import multiprocessing
from runner import SimulationArgs, print_results
from smc import smc_run
from splinecv import spline_run

if __name__ == '__main__':
    simulation_name = "nncv"
    M = 100
    #generator = mckeangenerator.SimpleGenerator()
    #payout = CallPayout(0.5)
    generator = lsv.LsvGenerator()
    payout = CallPayout(generator.market_vol.s0)

    args = SimulationArgs(generator, payout)
    arg_list = [args] * M
    pool_obj = multiprocessing.Pool(processes=4)
    result_list = pool_obj.map(nncv_run, arg_list)

    result_mc = np.array([result_list[i][0] for i in range(M)])
    result_mc_cv = np.array([result_list[i][1] for i in range(M)])
    result_mc_cv_mean = np.array([result_list[i][2] for i in range(M)])
    np.savez("data/"+simulation_name+"_"+args.generator.name+"_"+args.payout.name+".npz", result_mc, result_mc_cv, result_mc_cv_mean)
    print_results(simulation_name, result_mc, result_mc_cv, result_mc_cv_mean, args, sys.stdout)
    with open("data/report.txt", "a") as report_file:
        print_results(simulation_name, result_mc, result_mc_cv, result_mc_cv_mean, args, report_file)

