import numpy as np
import mckeangenerator
import lsv
from nncvmodelmulti import NNCVMultiModel
from payouts import CallPayout
from varminmodel import VarMinModel
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plot

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

h = 1 / 100
L = int(5 / h)
N = 1000
Nr = 5000
M = 100

r = 0
kappa = 9
v0 = 0.16*0.16
theta = 0.16*0.16
xi = 0.4
rho = -0.5

#generator = mckeangenerator.SimpleCorrGenerator(0)
#strike=0.5
generator = lsv.LsvGenerator(r, kappa, v0, theta, xi, rho)
strike = generator.market_vol.s0

payout = CallPayout(strike)
x, dw = generator.generate(Nr, L, h)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
tX = tf.constant(x[:, 1:, :], dtype=tf.float32)
model = NNCVMultiModel(x, dw, h, generator, payout)
current_loss = model.loss(model(tX))
print('Initial: loss=%2.8f' % current_loss)

epochs = range(100)
for epoch in epochs:
    for i in range(30):
        current_loss = model.trainstep(optimizer, tX)
    print('Epoch %2d: loss=%2.8f' % (epoch, current_loss))

result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_mc_cv_mean = np.zeros(M)

for j in range(M):
    X, deltaW = generator.generate(N, L, h)
    tX = tf.constant(X[:, 1:, :], dtype=tf.float32)
    b = generator.generate_diffusions(X, deltaW, h)
    w = model(tX).numpy()
    cv = np.sum(np.sum(w*b, axis=2), axis=1)
    result_mc_cv_mean[j] = np.mean(cv)
    f_T = payout(X)
    result_mc[j] = np.mean(f_T)
    result_mc_cv[j] = np.mean(f_T-cv)

np.savez("data/nncv_call.npz", result_mc, result_mc_cv)
print('MC: mean=%2.6f std=%2.6f' % (np.mean(result_mc), np.std(result_mc)))
print('MC-CV: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv), np.std(result_mc_cv)))
plot.boxplot([result_mc, result_mc_cv])
plot.show()


