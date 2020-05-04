from keras.losses import Loss
import tensorflow as tf
import numpy as np


class VarianceError(Loss):
    def __init__(self):
        super(VarianceError, self).__init__(name='variance_error')

    def call(self, y_true, y_pred):
        return tf.math.reduce_variance(y_true)
