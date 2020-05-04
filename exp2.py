import mckean
import numpy as np
import varminmodel as vm
import tensorflow as tf
import tensorflow.keras as keras;

h = 0.02
L = int(1 / h)
N = 1000
M = 100


X, deltaW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
model = vm.VarMinModel(X, deltaW)
v = model(X)
current_loss = model.loss(v)

learning_rate = 0.1

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
current_loss = model.loss(model(X))
print('Initial Loss: loss=%2.5f' % current_loss)
epochs = range(100)
for epoch in epochs:
    with tf.GradientTape() as t:
        t.watch(model.alpha)
        current_loss = model.loss(model(X))
    dAlpha = t.gradient(current_loss, [model.alpha])
    optimizer.apply_gradients(zip(dAlpha, [model.alpha]))
    print('Epoch %2d: loss=%2.5f' % (epoch, current_loss))
    print(model.alpha)


result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_mc_cv2 = np.zeros(M)


for j in range(M):
    X, deltaW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    cvF = np.sum(model(X).numpy() * mckean.bMat(X, deltaW), axis=1)
    result_mc[j] = np.mean(mckean.f(X[:, -1], X[:, -1]))
    result_mc_cv[j] = np.mean(mckean.f(X[:, -1], X[:, -1])-cvF)
    result_mc_cv2[j] = np.mean(cvF-mckean.f(X[:, -1], X[:, -1]))

print('MC: mean=%2.6f std=%2.6f' % (np.mean(result_mc), np.var(result_mc)))
print('MC-CV: mean=%2.6f std=%2.6f' % (np.mean(result_mc_cv), np.var(result_mc_cv)))
