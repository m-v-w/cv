import numpy as np
import mckean
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from varminmodel import VarMinModel


h = 0.02
L = int(1 / h)
N = 1000
M = 100

x, dw = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
m = VarMinModel(x, dw)

def fitFunc():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
