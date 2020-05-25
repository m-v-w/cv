import tensorflow as tf
import numpy as np
import mckean


class VarMinModel(object):
    def __init__(self, x, dw, h):
        # bv = mean(b(X(:,l)*ones(1,N),ones(N,1)*X(:,l)'),2);
        #         B(:,((l-1)*K+1):(l*K))=A.*(bv.*dW(:,l+1)*ones(1,K));
        self.K = 10
        N = x.shape[0]
        L = x.shape[1] - 1
        B = np.zeros((N, L))
        for l in range(L):
            xm = np.tile(x[:, l], (N, 1))
            B[:, l] = np.mean(mckean.b(xm.transpose(), xm), 1) * dw[:, l+1]
        self.b = B
        self.b = tf.constant(B, dtype=tf.float32)
        u, udW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
        self.f = tf.constant(np.mean(mckean.f(np.tile(x[:, -1], (N, 1)).transpose(), np.tile(u[:, -1], (N, 1))), 1), dtype=tf.float32)
        self.alpha = tf.Variable(tf.zeros([self.K]))
        base = np.zeros((N, L, self.K))
        for l in range(L):
            base[:, l, :] = mckean.genPoly(x[:, l], self.K) #* np.tile(self.b[:, 5], (self.K, 1)).transpose()
        self.tbase = tf.constant(base, dtype=tf.float32)

    def __call__(self, x):
        return tf.tensordot(self.tbase, self.alpha, axes=1)

    def loss(self, x):
        return tf.math.reduce_variance(tf.keras.backend.sum(x*self.b, axis=1)-self.f)

