import numpy as np
import mckeangenerator
from nncvmodel import NNCVModel
from nncvmodelmulti import NNCVMultiModel
from varminmodel import VarMinModel
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plot

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

h = 0.02
L = int(1 / h)
N = 1000
Nr = 5000
M = 100

generator = mckeangenerator.SimpleCorrGenerator(0)
x, dw = generator.generate(Nr, L, h)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
tX = tf.constant(x[:, 1:, :], dtype=tf.float32)
model = NNCVMultiModel(x, dw, h)
current_loss = model.loss(model(tX))
print('Initial: loss=%2.8f' % current_loss)

epochs = range(100)
for epoch in epochs:
    for i in range(30):
        current_loss = model.trainstep(optimizer, tX)
    print('Epoch %2d: loss=%2.8f' % (epoch, current_loss))

result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_cv = np.zeros(M)

for j in range(M):
    X, deltaW = generator.generate(N, L, h)
    tX = tf.constant(X[:, 1:], dtype=tf.float32)
    cvF = np.sum(model(tX).numpy() * mckean.bMat(X, deltaW), axis=1)
    result_cv[j] = np.mean(cvF)
    u, udW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    fT = np.mean(mckean.f(np.tile(X[:, -1], (N, 1)).transpose(), np.tile(u[:, -1], (N, 1))), 1)
    result_mc[j] = np.mean(fT)
    result_mc_cv[j] = np.mean(fT-cvF)

np.savez("data/nncv_f2.npz", result_mc, result_mc_cv)
print('MC: mean=%2.6f std=%2.6f' % (np.mean(result_mc), np.std(result_mc)))
print('MC-CV: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv), np.std(result_mc_cv)))
plot.boxplot([result_mc, result_mc_cv])


