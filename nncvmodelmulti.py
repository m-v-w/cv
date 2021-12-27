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
        B = generator.generate_diffusions(x, dw, h)
        d_x, d_w = generator.get_dimensions()
        self.b = tf.constant(B, dtype=tf.float32)
        self.models = []
        self.trainable_weights = []
        for j in range(L):
            m = tf.keras.Sequential([
                tf.keras.layers.Input(d_x),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),  # input shape required
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dense(d_x)
            ])
            self.trainable_weights.extend(m.trainable_weights)
            self.models.append(m)
        self.f = tf.constant(payout(x), dtype=tf.float32)
        self.loss_fn = VarianceError()
        self.pred_y = tf.zeros((N))

    def __call__(self, x):
        result = []
        L = x.shape[1]
        for l in range(L):
            result.append(self.models[l](x[:, l, :]))
        return tf.squeeze(tf.stack(result, axis=1))

    def trainstep(self, optimizer, tX):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(tX))
        dAlpha = t.gradient(current_loss, self.trainable_weights)
        optimizer.apply_gradients(zip(dAlpha, self.trainable_weights))
        return current_loss

    def loss(self, x):
        return self.loss_fn(K.sum(K.sum(x * self.b, axis=2), axis=1) - self.f, self.pred_y)

