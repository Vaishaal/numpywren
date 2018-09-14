import time
import random
from timeit import default_timer as timer
import string
import concurrent.futures as fs

from numpywren import compiler, job_runner, kernels
from numpywren.matrix import BigMatrix
from numpywren.alg_wrappers import cholesky, tsqr, gemm, qr, bdfac
import numpy as np
from numpywren.matrix_init import shard_matrix
from numpywren.algs import *
from numpywren.compiler import lpcompile, walk_program, find_parents, find_children
from numpywren import config
import pywren
import pywren.wrenconfig as wc

def test_cholesky():
    X = np.random.randn(64, 64)
    A = X.dot(X.T) + np.eye(X.shape[0])
    shard_size = 64
    shard_sizes = (shard_size, shard_size)
    A_sharded= BigMatrix("job_runner_test", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
    A_sharded.free()
    shard_matrix(A_sharded, A)
    program, meta =  cholesky(A_sharded)
    executor = fs.ProcessPoolExecutor(1)
    print("starting program")
    program.start()
    future = job_runner.lambdapack_run(program, timeout=60, idle_timeout=6)
    program.wait()
    program.free()
    L_sharded = meta["outputs"][0]
    L_npw = L_sharded.numpy()
    L = np.linalg.cholesky(A)
    assert(np.allclose(L_npw, L))
    print("great success!")


if __name__ == "__main__":
    pass
