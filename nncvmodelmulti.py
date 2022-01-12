import tensorflow as tf
import numpy as np
from keras import Input

import keras.backend as K

from mckeangenerator import IPathGenerator
from payouts import IPayout
from varminutils import VarianceError


class NNCVMultiModel(object):
    def __init__(self, x, dw, h, generator: IPathGenerator, payout: IPayout):
        N = x.shape[0]
        L = x.shape[1] - 1
        self.L = L
        B = generator.generate_diffusions(x, dw, h)
        d_x, d_w = generator.get_dimensions()
        self.d = d_x
        if d_x == 1:
            self.b = tf.constant(B[:, :, 0], dtype=tf.float32)
        else:
            self.b = tf.constant(B, dtype=tf.float32)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((L, d_x)),
            tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu),
            tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu),
            tf.keras.layers.LocallyConnected1D(d_x, 1)
        ])
        self.f = tf.constant(payout(x), dtype=tf.float32)
        self.loss_fn = VarianceError()
        self.pred_y = tf.zeros((N))

    def __call__(self, x, training=False):
        return self.model(x, training=training)

    def trainstep(self, optimizer, tX):
        with tf.GradientTape() as t:
            current_loss = self.loss(self.model(tX, training=True))
        dAlpha = t.gradient(current_loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(dAlpha, self.model.trainable_weights))
        return current_loss

    def loss(self, x):
        if self.d == 1:
            return self.loss_fn(K.sum(x * self.b, axis=1) - self.f, self.pred_y)
        else:
            return self.loss_fn(K.sum(K.sum(x * self.b, axis=2), axis=1) - self.f, self.pred_y)


def loss_std(x, b, f):
    return tf.math.reduce_std(f - tf.math.reduce_sum(tf.math.reduce_sum(x * b, axis=2), axis=1))


def normal_kernel(x):
    return tf.math.exp(-0.5*tf.math.square(x))


def flatten_time(x, h):
    Nr = x.shape[0]
    L = x.shape[1]
    return attach_time(x, h).reshape((Nr*L, x.shape[2]+1))


def attach_time(x, h):
    Nr = x.shape[0]
    L = x.shape[1]
    times = np.tile(h*np.array(range(L)), (Nr, 1)).reshape((Nr, L, 1))
    return np.concatenate((x, times), axis=2)
