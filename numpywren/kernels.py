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

def gemm(A, B, *args, **kwargs):
    if (kwargs.get('transpose_A', False)):
        A = A.T
    if (kwargs.get('transpose_B', False)):
        B = B.T
    return A.dot(B)

def qr_trailing_update(Q0, Q1, S1, S2, *args, **kwargs):
    pass

def trsm(x, y, lower=False, right=True, *args, **kwargs):
    return scipy.linalg.blas.dtrsm(1.0, x.T, y, lower=lower, side=int(right))
