import tensorflow as tf
import numpy as np
import mckean


class VarMinModel(object):
    def __init__(self, x, dw):
        # bv = mean(b(X(:,l)*ones(1,N),ones(N,1)*X(:,l)'),2);
        #         B(:,((l-1)*K+1):(l*K))=A.*(bv.*dW(:,l+1)*ones(1,K));
        self.K = 5
        N = x.shape[0]
        L = x.shape[1] - 1;
        B = np.zeros((N, L))
        for l in range(L):
            B[:, l] = np.mean(mckean.b(np.ones(N) * x[:, l], x[:, l]) * dw[:, l+1])

        self.b = tf.constant(B.transpose())
        self.f = tf.constant(np.mean(mckean.f(x[:, -1], x[:, -1])))
        self.alpha = tf.Variable(tf.ones([self.K]))

    def __call__(self, x):
        L = x.shape[1] - 1
        N = x.shape[0]
        result = tf.zeros([N, L])
        for l in range(L):
            base = mckean.genPoly(x[:, l], self.K)
            result[:, l] = tf.matmul(base, self.alpha)
        return result

    def loss(self, x):
        return tf.reduce_variance(tf.sum(x*self.b, 1)-self.f)

