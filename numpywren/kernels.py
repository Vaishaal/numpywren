import numpy as np
import scipy.linalg
import boto3
import os
import sys

def add(*args, **kwargs):
    out = np.zeros(args[0].shape)
    for a in args:
        out += a
    return out

def get_shared_so(so_name):
    if (not os.path.isfile(f"/tmp/{so_name}")):
        with open(f"/tmp/{so_name}", "wb+") as f:
            client = boto3.client('s3')
            bstream = client.get_object(Bucket="numpywrenpublic", Key=f"shared_sos/{so_name}")["Body"].read()
            f.write(bstream)



def slow_qr(x):
    get_shared_so("dlarft.cpython-36m-x86_64-linux-gnu.so")
    import sys
    import scipy.linalg
    sys.path.insert(0, "/tmp/")
    lst = os.listdir("/tmp/")
    for elem in lst:
        if "condaruntime" in elem:
            conda_root = elem
            break

    ld_path = os.path.join("/tmp/", conda_root, "/lib/")
    os.environ["LD_LIBRARY_PATH"] = ld_path
    import dlarft
    qr, tau, work, info = scipy.linalg.lapack.dgeqrf(a=x)
    r = np.triu(qr)
    k = min(x.shape[0], x.shape[1])
    t = np.zeros((k, k), order='F')
    v = np.tril(qr)
    v = v[:,:k]
    idxs = np.diag_indices(k)
    v[idxs] = 1
    idxs = np.diag_indices(min(v.shape[0], v.shape[1]))
    r = r[:r.shape[1],:]
    dlarft.dlarft(direct='F',storev='C', n=v.shape[0], k=k, v=v, ldv=v.shape[0], tau=tau, t=t, ldt=t.shape[0])
    return v,t,r

def fast_qr(x):
    if (os.path.exists("/tmp/dgqert3.cpython-36m-x86_64-linux-gnu.so")):
        os.remove("/tmp/dgqert3.cpython-36m-x86_64-linux-gnu.so")
    get_shared_so("dgqert3.cpython-36m-x86_64-linux-gnu.so")
    sys.path.insert(0, "/tmp/")
    import scipy.linalg
    conda_root = ""
    lst = os.listdir("/tmp/")
    for elem in lst:
        if "condaruntime" in elem:
            conda_root = elem
            break

    ld_path = os.path.join("/tmp/", conda_root, "/lib/python3.6/site-packages/numpy/linalg/")
    import dgqert3
    m = x.shape[0]
    n = x.shape[1]
    k = min(x.shape[0], x.shape[1])
    transposed = False
    if (n > m):
        return slow_qr(x)
    x = x.copy(order='F')
    t = np.zeros((x.shape[1], x.shape[1]), order='F')
    dgqert3.dgeqrt3(m=x.shape[0], n=x.shape[1], a=x, t=t, info=0)
    r = np.triu(x)
    v = np.triu(x.T).T
    idxs = np.diag_indices(min(v.shape[0], v.shape[1]))
    v = v[:,:k]
    v[idxs] = 1
    r = r[:r.shape[1],:]
    return v,t,r

def qr_factor(*blocks, **kwargs):
    ins = np.vstack(blocks)
    v,t,r = fast_qr(ins)
    return v,t,r

def qr_leaf(V, T, S0, *args, **kwargs):
    return S0 - (V @ T.T @ V.T @ S0)

def identity(X, *args, **kwargs):
    return X

def qr_trailing_update(V, T, S0, S1=None, *args, **kwargs):
    if (S1 is None):
        return qr_leaf(V, T, S0), np.zeros(S0.shape)
    V = V[-S0.shape[0]:]
    W = T.T @ (S0 + V.T @ S1)
    S01 = S0 - W
    S11 = S1 - V.dot(W)
    return S01, S11

def syrk(s, x, y, *args, **kwargs):
    return s - x.dot(y.T)

def chol(x, *args, **kwargs):
    return np.linalg.cholesky(x)

def mul(x, y, *args, **kwargs):
    return x * y

def identity(x, *args, **kwargs):
    return x

def gemm(A, B, *args, **kwargs):
    if (kwargs.get('transpose_A', False)):
        A = A.T
    if (kwargs.get('transpose_B', False)):
        B = B.T
    return A.dot(B)


def trsm(x, y, lower=False, right=True, *args, **kwargs):
    return scipy.linalg.blas.dtrsm(1.0, x.T, y, lower=lower, side=int(right))

if __name__ == "__main__":
    x = np.random.randn(4,4)
    print(fast_qr(x))
    print(slow_qr(x))


