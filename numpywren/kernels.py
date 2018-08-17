import numpy as np
import scipy.linalg
import boto3
import os

def add(*args, **kwargs):
    out = np.zeros(args[0].shape)
    for a in args:
        out += a
    return out

def fast_qr(x):
    if (not os.path.isfile("/tmp/dgqert3.cpython-36m-x86_64-linux-gnu.so")):
        with open("/tmp/dgqert3.cpython-36m-x86_64-linux-gnu.so", "wb+") as f:
            client = boto3.client('s3')
            bstream = client.get_object(Bucket="numpywrenpublic", Key="shared_sos/dgqert3.cpython-36m-x86_64-linux-gnu.so")["Body"].read()
            f.write(bstream)
    import sys
    sys.path.insert(0, "/tmp/")
    import dgqert3
    x = x.copy()
    t = np.zeros((x.shape[1], x.shape[1]), order='F')
    dgqert3.dgeqrt3(x.shape[0], x.shape[1], a=x, t=t, info=0)
    r = np.triu(x)
    v = np.triu(x.T).T
    idxs = np.diag_indices(v.shape[0])
    v[idxs] = 1
    return v,t,r






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
