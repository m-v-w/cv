import sys
import numpy as np
import tensorflow as tf
from runner import SimulationArgs


model = tf.keras.models.load_model("data/nncv_model_lsv_call", compile=False)

def nncv_run(args: SimulationArgs):
    N = args.N
    L = args.L
    generator = args.generator
    h = args.h
    payout = args.payout
    d_x, d_w = generator.get_dimensions()
    X, deltaW = generator.generate(N, L, h)
    tX = tf.constant(X[:, 1:, :], dtype=tf.float32)
    b = generator.generate_diffusions(X, deltaW, h)
    w = model(tX).numpy()
    if len(w.shape) > 2:
        cv = np.sum(w * b, axis=(1, 2))
    else:
        cv = np.sum(w * b[:, :, 0], axis=1)
    result_mc_cv_mean = np.mean(cv)
    f_T = payout(X)
    result_mc = np.mean(f_T)
    result_mc_cv = np.mean(f_T-cv)
    print('smc={smc:.6f}, cv={cv:.6f}, cv_mean={cv_mean:.6f}'.format(smc=result_mc, cv=result_mc_cv, cv_mean=result_mc_cv_mean))
    return result_mc, result_mc_cv, result_mc_cv_mean
