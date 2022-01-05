import tensorflow as tf
import pydot
from tensorflow.keras.utils import plot_model


m.add(tf.keras.layers.Input((500, 2)))
m.add(tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu))
m.add(tf.keras.layers.LocallyConnected1D(100, 1, activation=tf.nn.relu))
m.add(tf.keras.layers.LocallyConnected1D(2, 1))
m.build()
print(m.output_shape)
tf.keras.utils.plot_model(m, expand_nested=True, show_shapes=True)
