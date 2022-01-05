import sys
from datetime import datetime

import numpy as np
import mckeangenerator
import lsv
from nncvmodelmulti import NNCVMultiModel
from payouts import CallPayout
from runner import SimulationArgs, print_results
from varminmodel import VarMinModel
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plot

from varminutils import VarianceError

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

M = 100
#generator = mckeangenerator.SimpleGenerator()
#payout = CallPayout(0.5)
generator = lsv.LsvGenerator()
payout = CallPayout(generator.market_vol.s0)

args = SimulationArgs(generator, payout)
N = args.N
Nr = args.Nr
L = args.L
h = args.h
x, dw = generator.generate(Nr, L, h)
diffusion_matrix = generator.generate_diffusions(x, dw, h)

train_data = tf.data.Dataset.from_tensor_slices((x[:, 1:, :].astype(np.float32), diffusion_matrix.astype(np.float32), payout(x).astype(np.float32)))
train_batches = train_data.shuffle(666).batch(100)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.03,
    decay_steps=75,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
d_x, d_w = generator.get_dimensions()
model = tf.keras.Sequential([
    tf.keras.layers.Input((L, d_x)),
    tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu),
    tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu),
    tf.keras.layers.LocallyConnected1D(d_x, 1)
])


def loss(x, b, f):
    return tf.math.reduce_variance(K.sum(K.sum(x * b, axis=2), axis=1) - f)

#current_loss = model.loss(model(train_data.take(1)))
#print('Initial: loss=%.10f' % current_loss)
log_dir = "logs/gradient_tape/" + datetime.now().strftime("%Y%m%d-%H%M%S") +"/train"
train_summary_writer = tf.summary.create_file_writer(log_dir)

epochs = range(100)
for epoch in epochs:
    for (batch_x, batch_b, batch_f) in train_batches:
        with tf.GradientTape() as t:
            current_loss = loss(model(batch_x, training=True), batch_b, batch_f)
        dAlpha = t.gradient(current_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(dAlpha, model.trainable_weights))
        train_loss(current_loss)
    current_learning_rate = optimizer._decayed_lr(tf.float32)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('learning_rate', current_learning_rate, step=epoch)
    print('Epoch %d: loss=%.10f lr=%.10f' % (epoch+1, current_loss, current_learning_rate))
    train_loss.reset_states()
    #if current_loss < 0.00000010:
        #break

model.save("data/nncv_model_"+args.generator.name+"_"+args.payout.name+"_%d" % args.L)
result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_mc_cv_mean = np.zeros(M)

for j in range(M):
    X, deltaW = generator.generate(N, L, h)
    tX = tf.constant(X[:, 1:, :], dtype=tf.float32)
    b = generator.generate_diffusions(X, deltaW, h)
    w = model(tX).numpy()
    if len(w.shape) > 2:
        cv = np.sum(w * b, axis=(1, 2))
    else:
        cv = np.sum(w * b[:, :, 0], axis=1)
    result_mc_cv_mean[j] = np.mean(cv)
    f_T = payout(X)
    result_mc[j] = np.mean(f_T)
    result_mc_cv[j] = np.mean(f_T-cv)
    print('{l:d} smc={smc:.6f}, cv={cv:.6f}, cv_mean={cv_mean:.6f}'.format(l=j, smc=result_mc[j], cv=result_mc_cv[j], cv_mean=result_mc_cv_mean[j]))

np.savez("data/nncv_"+args.generator.name+"_"+args.payout.name+"_%d.npz" % args.L, result_mc, result_mc_cv, result_mc_cv_mean)
print_results("nncv", result_mc, result_mc_cv, result_mc_cv_mean, args, sys.stdout)
with open("data/report.txt", "a") as report_file:
    print_results("nncv", result_mc, result_mc_cv, result_mc_cv_mean, args, report_file)
