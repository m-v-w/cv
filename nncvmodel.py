import tensorflow as tf
import numpy as np
from keras import Input

import mckean
import keras.backend as K
from varminutils import VarianceError


class NNCVModel(object):
    def __init__(self, x, dw):
        N = x.shape[0]
        L = x.shape[1] - 1
        B = mckean.bMat(x, dw)
        self.b = tf.constant(B, dtype=tf.float32)
        lIn = Input(shape=(L,))

        self.models = []
        self.trainable_weights = []
        for j in range(L):
            m = tf.keras.Sequential([
                tf.keras.layers.Input(1),
                tf.keras.layers.Dense(50, activation=tf.nn.relu),  # input shape required
                tf.keras.layers.Dense(50, activation=tf.nn.relu),
                tf.keras.layers.Dense(1)
            ])
            self.trainable_weights.extend(m.trainable_weights)
            self.models.append(m)

        xm = np.tile(x[:, -1], (N, 1))
        self.f = tf.constant(np.mean(mckean.f(xm.transpose(), xm), 1), dtype=tf.float32)
        self.loss_fn = VarianceError()
        self.pred_y = tf.zeros((N))

    def __call__(self, x):
        result = []
        L = x.shape[1]
        for l in range(L):
            result.append(self.models[l](x[:, l:(l+1)]))
        return tf.squeeze(tf.stack(result, axis=1))

    def trainstep(self, optimizer, tX):
        with tf.GradientTape() as t:
            current_loss = self.loss(self(tX))
        dAlpha = t.gradient(current_loss, self.trainable_weights)
        optimizer.apply_gradients(zip(dAlpha, self.trainable_weights))
        return current_loss

    def loss(self, x):
        return self.loss_fn(K.sum(x * self.b, axis=1) - self.f, self.pred_y)

