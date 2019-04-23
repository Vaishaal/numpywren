import os
import sys
import random
import fcntl
import boto3
import numpy as np
import scipy.linalg
from . import utils



NUM_SO_SHARDS = 100
SO_TAIL = f"cpython-36m-x86_64-linux-gnu.so"


def add_matrices(*args, **kwargs):
    out = np.zeros(args[0].shape)
    for a in args:
        out += a
    return out

def get_shared_so(so_name):
    pid = os.getpid()
    out_str = f"/tmp/{so_name}.{SO_TAIL}"
    lock = open("/tmp/so_lock", "a")
    fcntl.lockf(lock, fcntl.LOCK_EX)
    if (os.path.exists(out_str)):
        return
    shard = random.randint(1,NUM_SO_SHARDS)
    if (shard > 0):
        shard_str = str(shard)
    else:
        shard_str = ""
    print(f"Fetching lapack/{so_name}.{SO_TAIL}_{shard_str}")
    client = boto3.client('s3')
    bstream = utils.get_object_with_backoff(client, bucket="numpywrenpublic", key=f"lapack/{so_name}.{SO_TAIL}_{shard_str}")
    with open(out_str, "wb+") as f:
        f.write(bstream)
    fcntl.lockf(lock, fcntl.LOCK_UN)
    lock.close()


def banded_to_bidiagonal(x):
    get_shared_so("dgbbrd.so")
    sys.path.insert(0, "/tmp/")
    import dgbbrd
    shard_size = x[0].shape[0]
    x_size = shard_size * (len(x) - 1) + x[-1].shape[0]
    band_size = shard_size - 1
    num_packed_rows = 2 * band_size + 1
    packed_x = np.zeros([num_packed_rows, x_size])
    import time
    start = time.time()
    for i, block in enumerate(x):
        for j in range(block.shape[0]):
            packed_x[num_packed_rows - j - shard_size:num_packed_rows - j, i * shard_size + j] = block[:, j]
        print(i, time.time() - start)
    diag_out = np.zeros([x_size])
    offdiag_out = np.zeros([x_size - 1])
    work = np.zeros([2 * x_size])
    print("A")
    start = time.time()
    dgbbrd.dgbbrd(vect="N", m=x_size, n=x_size, ncc=0, kl=band_size, ku=band_size, ab=packed_x, ldab=num_packed_rows, d=diag_out, e=offdiag_out, q=None, ldq=1, pt=None, ldpt=1, c=None, ldc=1, work=work, info=0)
    print(time.time() - start)
    return diag_out, offdiag_out

def slow_qr(x):
    get_shared_so("dlarft")
    import sys
    import scipy.linalg
    sys.path.insert(0, "/tmp/")
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
    get_shared_so("dgeqrt3")
    sys.path.insert(0, "/tmp/")
    import dgeqrt3
    m = x.shape[0]
    n = x.shape[1]
    k = min(x.shape[0], x.shape[1])
    transposed = False
    if (n > m):
        return slow_qr(x)
    x = x.copy(order='F')
    t = np.zeros((x.shape[1], x.shape[1]), order='F')
    dgeqrt3.dgeqrt3(m=x.shape[0], n=x.shape[1], a=x, t=t, info=0)
    r = np.triu(x)
    v = np.triu(x.T).T
    idxs = np.diag_indices(min(v.shape[0], v.shape[1]))
    v = v[:,:k]
    v[idxs] = 1
    r = r[:r.shape[1],:]
    return v,t,r

def fast_qr_triangular(x0, x1):
    get_shared_so("dtpqrt")
    sys.path.insert(0, "/tmp/")
    import dtpqrt
    m = x0.shape[0]
    n = x0.shape[1]
    k = min(x0.shape[0], x0.shape[1])
    transposed = False
    x0 = x0.copy(order='F')
    x1 = x1.copy(order='F')
    t = np.zeros((x0.shape[1], x0.shape[1]), order='F')
    work = np.zeros([32* x0.shape[0]])
    dtpqrt.dtpqrt(m=x0.shape[0], n=x0.shape[1], nb=min(n, 32), l=x0.shape[0], a=x0, b=x1, t=t, work=work, info=0)
    r = np.triu(x0)
    v = np.triu(x1.T).T
    idxs = np.diag_indices(min(v.shape[0], v.shape[1]))
    v[idxs] = 1
    return v,t,r


def qr_factor(*blocks, **kwargs):
    ins = np.vstack(blocks)
    v,t,r = fast_qr(ins)
    return v,t,r

def qr_factor_triangular(x0, x1, **kwargs):
    v,t,r = fast_qr_triangular(x0, x1)
    return v,t,r


def _qr_flops(*blocks):
    ins = np.vstack(blocks)
    m = ins.shape[0]
    n = ins.shape[1]
    return 2*m*n*n - (2*n**3)/3

qr_factor.flops = _qr_flops

def lq_factor(*blocks, **kwargs):
    if len(blocks) == 2:
        assert(blocks[0].shape[0] == blocks[1].shape[0])
    ins = np.hstack(blocks)
    v,t,r = fast_qr(ins.T)
    return v.T,t.T,r.T

lq_factor.flops = _qr_flops

def lq_leaf(V, T, S0, *args, **kwargs):
    # (I - VTV)^{T}*S
    val = S0 - S0 @ V.T @ T.T @ V
    return val


def qr_leaf(V, T, S0, *args, **kwargs):
    # (I - VTV)^{T}*S
    #val = S0 - (V @ T.T @ (V.T @ S0))
    val = S0 - (V.T @ S0)
    return val

def _qr_leaf_flops(V, T, S0):
    c0 = V.shape[0] * S0.shape[0] * S0.shape[1]
    c1 = T.shape[0] * V.shape[0] * S0.shape[1]
    c2 = V.shape[0] * T.shape[0] * T.shape[1]
    return c0 + c1 + c2 + S0.shape[0]*S0.shape[1]

qr_leaf.flops = _qr_leaf_flops
lq_leaf.flops = _qr_leaf_flops

def identity(X, *args, **kwargs):
    return X

def trsm_sub(L, S, x):
    return scipy.linalg.solve_triangular(L, x - S)

def qr_trailing_update(V, T, S0, S1, *args, **kwargs):
    if (S1 is None):
        return qr_leaf(V, T, S0), np.zeros(S0.shape)
    V = V[-S0.shape[0]:]
    W = T.T @ (S0 + V.T @ S1)
    S01 = S0 - W
    S11 = S1 - V.dot(W)
    return S01, S11

def _qr_trailing_flops(V, T, S0, S1):
    M, N = V.shape
    c0 = M * S1.shape[0] * S1.shape[1]
    c1 = T.shape[0]*T.shape[1]*S0.shape[1]
    return 2*c1 + c0 + T.shape[0]*T.shape[1]

qr_trailing_update.flops = _qr_trailing_flops


def lq_trailing_update(V, T, S0, S1=None, *args, **kwargs):
    if (S1 is None):
        return lq_leaf(V, T, S0), np.zeros(S0.shape)
    V = V[:, -S0.shape[0]:]
    W = (S0 + S1 @ V.T) @ T.T
    S01 = S0 - W
    S11 = S1 - W.dot(V)
    assert(S0.shape == S01.shape)
    assert(S1.shape == S11.shape)
    return S01, S11

lq_trailing_update.flops = _qr_trailing_flops

def syrk(s, x, y, *args, **kwargs):
    if (np.allclose(x, 0) or np.allclose(y, 0)):
        return s
    return s - x.dot(y.T)

def _syrk_flops(s, x, y):
    m = x.shape[0]
    n = x.shape[1]
    z = y.shape[1]
    return 2*m*n*z + m*z

syrk.flops  = _syrk_flops

def chol(x, *args, **kwargs):
    return np.linalg.cholesky(x)

def _chol_flops(x):
    return (x.shape[0]**3)/3

chol.flops = _chol_flops

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

def _gemm_flops(A,B):
    m,n = A.shape
    k = B.shape[1]
    return 2*m*n*k

gemm.flops = _gemm_flops


def trsm(x, y, lower=False, right=True, *args, **kwargs):
    if np.allclose(y, 0):
        return np.zeros((x.shape[1], y.shape[0]))
    return scipy.linalg.blas.dtrsm(1.0, x.T, y, lower=lower, side=int(right))

def _trsm_flops(x, y):
    if (len(y.shape) == 0):
        return x.shape[0]*x.shape[1]
    else:
        return x.shape[0]*x.shape[1]*y.shape[1]


if __name__ == "__main__":
    x = np.random.randn(4,4)
    y = np.random.randn(4,4)
    x = np.triu(x)
    y = np.triu(y)
    print(fast_qr(x,y))

