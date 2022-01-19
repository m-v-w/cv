import numpy as np

import mckeangenerator
import lsv
from payouts import SquaredPayout, CallPayout
from ploycv import poly_run
from runner import SimulationArgs
from matplotlib import pyplot as plot
from splinecv import spline_run

simulation_name = "spline"
#generator = mckeangenerator.SimpleGenerator()
#generator = mckeangenerator.SimpleCorrGenerator(-0.5)
#payout = TrigonometricPayout(generator)
#payout = CallPayout(0.5)
#payout = SquaredPayout()
generator = lsv.LsvGenerator()
payout = CallPayout(generator.market_vol.s0)

args = SimulationArgs(generator, payout)
result_mc, result_cv, result_cv_mean, delta_phi, coeff = spline_run(args, True)
for i in range(5):
    plot.plot(delta_phi[i, 1:, 0])
plot.savefig("plots/delta_" + simulation_name + "_" + generator.name + ".eps", format='eps')
plot.savefig("plots/delta_" + simulation_name + "_" + generator.name + ".png", format='png')
plot.show()
plot.boxplot(coeff)
plot.savefig("plots/coeff_" + simulation_name + "_" + generator.name + ".eps", format='eps')
plot.savefig("plots/coeff_" + simulation_name + "_" + generator.name + ".png", format='png')
plot.show()
plot.plot(np.max(coeff, axis=1))
plot.show()
print(np.argmax(coeff, axis=0))
print("done")
