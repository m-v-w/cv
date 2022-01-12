import sys
from datetime import datetime

import numpy as np
import mckeangenerator
import lsv
from nncvmodelmulti import loss_std, normal_kernel, flatten_time, attach_time
from payouts import CallPayout, TrigonometricPayout
from runner import SimulationArgs, print_results
from varminmodel import VarMinModel
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plot

from varminutils import VarianceError

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
np.random.seed(1)
M = 100
#generator = mckeangenerator.SimpleGenerator()
#payout = TrigonometricPayout(0.02, generator)  # TODO adjust h
#payout = CallPayout(0.5)

generator = lsv.LsvGenerator()
payout = CallPayout(generator.market_vol.s0)

args = SimulationArgs(generator, payout)
N = args.N
Nr = args.Nr
L = args.L
h = args.h
batch_size = 100
activation = 'sigmoid'
output_activation = 'linear'
hidden_nodes = 1000
x, dw, diffusion_delta = generator.generate(Nr, L, h, return_diffusion_delta=True)
test_x, test_dw, test_diffusion_delta = generator.generate(Nr, L, h, return_diffusion_delta=True)
train_data = tf.data.Dataset.from_tensor_slices((attach_time(x, h).astype(np.float32), diffusion_delta.astype(np.float32), payout(x).astype(np.float32)))
test_data = tf.data.Dataset.from_tensor_slices((attach_time(test_x, h).astype(np.float32), test_diffusion_delta.astype(np.float32), payout(test_x).astype(np.float32))).batch(batch_size)
train_batches = train_data.shuffle(Nr).batch(batch_size)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
initial_loss = tf.keras.metrics.Mean('initial_loss', dtype=tf.float32)
for (x_test, b_test, f_test) in test_data:
    l = loss_std(tf.zeros(b_test.shape), b_test, f_test)
    initial_loss(l)
print("Initial test lost %.10f" % initial_loss.result())
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=round(3*Nr/batch_size),
    decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
d_x, d_w = generator.get_dimensions()
model = tf.keras.Sequential([
    tf.keras.layers.Input((L+1, d_x+1)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_nodes, activation=activation)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_nodes, activation=activation)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(d_x, activation='linear'))
])
log_dir = "logs/gradient_tape/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-"+args.generator.name+"-"+args.payout.name+"-%d" % args.L
train_summary_writer = tf.summary.create_file_writer(log_dir)
with train_summary_writer.as_default():
    tf.summary.text('activation', activation, step=0)
    tf.summary.text('output_activation', output_activation, step=0)
epochs = range(100)
for epoch in epochs:
    # subset = train_batches.shuffle(Nr).take(int(50/4))
    for (batch_x, batch_b, batch_f) in train_batches:
        with tf.GradientTape() as t:
            current_loss = loss_std(model(batch_x, training=True), batch_b, batch_f)
        dAlpha = t.gradient(current_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(dAlpha, model.trainable_weights))
        train_loss(current_loss)
    for (x_test, b_test, f_test) in test_data:
        l = loss_std(model(x_test), b_test, f_test)
        test_loss(l)
    current_learning_rate = optimizer._decayed_lr(tf.float32)
    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('learning_rate', current_learning_rate, step=epoch)
    print('Epoch %d: loss=%.10f test_loss=%.10f lr=%.10f' % (epoch+1, train_loss.result(), test_loss.result(), current_learning_rate))
    train_loss.reset_states()
    test_loss.reset_states()

model.save("data/nncv_model_"+args.generator.name+"_"+args.payout.name+"_%d" % args.L)
result_mc = np.zeros(M)
result_mc_cv = np.zeros(M)
result_mc_cv_mean = np.zeros(M)

for j in range(M):
    X, _, diffusion_delta = generator.generate(N, L, h, return_diffusion_delta=True)
    prediction = model.predict(attach_time(X, h))
    #tX = tf.constant(X, dtype=tf.float32)
    b = diffusion_delta
    w = prediction # model(tX).numpy()
    cv = np.sum(w * b, axis=(1, 2))
    result_mc_cv_mean[j] = np.mean(cv)
    f_T = payout(X)
    result_mc[j] = np.mean(f_T)
    result_mc_cv[j] = np.mean(f_T-cv)
    print('{l:d} smc={smc:.6f}, cv={cv:.6f}, cv_mean={cv_mean:.6f}'.format(l=j, smc=result_mc[j], cv=result_mc_cv[j], cv_mean=result_mc_cv_mean[j]))

np.savez("data/nncv_"+args.generator.name+"_"+args.payout.name+"_%d.npz" % args.L, result_mc, result_mc_cv, result_mc_cv_mean)
print_results("nncv", result_mc, result_mc_cv, result_mc_cv_mean, args, sys.stdout)
with open("data/report.txt", "a") as report_file:
    print_results("nncv", result_mc, result_mc_cv, result_mc_cv_mean, args, report_file)
