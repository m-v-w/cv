import numpy as np
from matplotlib import pyplot as plot


nncvnpz = np.load("data/nncv_f1.npz")
#Matlab output: refV towerReg towerRegV2 varMinSpline2
regr = np.loadtxt("data/matlab_f1.csv",delimiter=',')
outfile = "plots/nncv_f1.eps"


result_mc = nncvnpz['arr_0']
result_nncv = nncvnpz['arr_1']
regr_mc = regr[:, 0]
regr_tower_poly = regr[:, 1]
regr_tower_spline = regr[:, 2]
regr_varmin_spline = regr[:, 3]
plotData = [result_mc, result_nncv, regr_tower_poly, regr_tower_spline, regr_varmin_spline]
labels = ['SMC', 'Neural Network', 'Polynom', 'Spline', 'Spline var-min']

len = plotData.__len__()
plot.boxplot(plotData)
plot.xticks(range(1,len+1), labels)
plot.savefig(outfile)
plot.show()

for i in range(len):
    print('%s: mean=%2.6f std=%2.6f' % (labels[i], np.mean(plotData[i]), np.std(plotData[i])))
print('Tower-Poly: mean=%2.6f std=%2.6f' % (np.mean(regr_tower_poly), np.std(regr_tower_poly)))
