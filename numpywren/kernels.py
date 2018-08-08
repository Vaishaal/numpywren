import numpy as np
import scipy.linalg

def add(*args, **kwargs):
    out = np.zeros(args[0].shape)
    for a in args:
        out += a
    return out

def qr_factor(*blocks, **kwargs):
    ins = np.vstack(blocks)
    out = np.linalg.qr(ins)
    print("IN SHAPE", ins.shape)
    print("OUT Q SHAPE", out[0].shape)
    print("OUT R SHAPE", out[1].shape)
    return out

def syrk(s, x, y, *args, **kwargs):
    return s - x.dot(y.T)

def chol(x, *args, **kwargs):
    return np.linalg.cholesky(x)

def mul(x, y, *args, **kwargs):
    return x * y

def trsm(x, y, lower=False, right=True, *args, **kwargs):
    return scipy.linalg.blas.dtrsm(1.0, x.T, y, lower=lower, side=int(right))
