import numpy as np
import mckean
from nncvmodel import NNCVModel
from varminmodel import VarMinModel
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plot

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

h = 0.02
L = int(1 / h)
N = 1000
M = 100

x, dw = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
m = VarMinModel(x, dw)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
tX = tf.constant(x[:, 1:], dtype=tf.float32)
model = NNCVModel(x, dw)
current_loss = model.loss(model(tX))
print('Initial: loss=%2.5f' % current_loss)
#tf.math.reduce_variance(tf.keras.backend.sum(x*self.b, axis=1)-self.f)

epochs = range(100)
for epoch in epochs:
    for i in range(30):
        current_loss = model.trainstep(optimizer, tX)
    print('Epoch %2d: loss=%2.5f' % (epoch, current_loss))


result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_cv = np.zeros(M)

for j in range(M):
    X, deltaW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    tX = tf.constant(X[:, 1:], dtype=tf.float32)
    cvF = np.sum(model(tX).numpy() * mckean.bMat(X, deltaW), axis=1)
    result_cv[j] = np.mean(cvF)
    result_mc[j] = np.mean(mckean.f(X[:, -1], X[:, -1]))
    result_mc_cv[j] = np.mean(mckean.f(X[:, -1], X[:, -1])-cvF)

print('MC: mean=%2.6f std=%2.6f' % (np.mean(result_mc), np.std(result_mc)))
print('MC-CV: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv), np.std(result_mc_cv)))
plot.boxplot([result_mc, result_mc_cv])


