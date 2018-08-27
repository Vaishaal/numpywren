from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner
from numpywren import compiler, frontend, job_runner, kernels
from numpywren.matrix_init import shard_matrix
import numpywren as npw

import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import os
import numpywren
import boto3


def matmul(A:BigMatrix, B:BigMatrix, M:int, N:int, K:int, bfac:int, Temp:BigMatrix, Out:BigMatrix):
    tree_depth = ceiling(log(K)/log(bfac))
    for i in range(0, M):
        for j in range(0, N):
            for k in range(0, K):
                Temp[i, j, k, 0] = gemm(A[i, k], B[k, j])

    for i in range(0, M):
        for j in range(0, N):
            with reducer(expr=Temp[i, j, k, 0], var=k, start=0, end=K, b_fac=bfac) as r:
                Temp[i, j, k, r.level] = add(*r.reduce_args)
                r.reduce_next(Temp[i, j, k, r.level])
            Out[i, j] = identity(Temp[i, j, 0, tree_depth - 1])

class MatmulTest(unittest.TestCase):
    def test_matmul(self):
        size = 4
        shard_size = 2
        np.random.seed(0)
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        C = np.dot(A, B)

        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("matmul_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        B_sharded= BigMatrix("matmul_test_B", shape=B.shape, shard_sizes=shard_sizes, write_header=True)
        B_sharded.free()
        shard_matrix(B_sharded, B)
        Temp = BigMatrix("matmul_test_Temp", shape=[A.shape[0], B.shape[1], B.shape[0], 100], shard_sizes=[A_sharded.shard_sizes[0], B_sharded.shard_sizes[1], 1, 1], write_header=True)
        C_sharded= BigMatrix("matmul_test_C", shape=C.shape, shard_sizes=shard_sizes, write_header=True)

        b_fac = 2
        config = npw.config.default()
        compiled_matmul = frontend.lpcompile(matmul)
        program = compiled_matmul(A_sharded, B_sharded, A_sharded.num_blocks(0), A_sharded.num_blocks(1), B_sharded.num_blocks(1), b_fac, Temp, C_sharded)
        program_executable = lp.LambdaPackProgram(program, config=config)
        program_executable.start()
        job_runner.lambdapack_run(program_executable, pipeline_width=1, idle_timeout=5, timeout=60)
        executor = fs.ThreadPoolExecutor(1)
        all_futures = [executor.submit(job_runner.lambdapack_run, program_executable, pipeline_width=1, idle_timeout=5, timeout=60)]
        program_executable.wait()
        program_executable.free()
        C_remote = C_sharded.numpy()
        assert(np.allclose(C, C_remote))

a = MatmulTest()
a.test_matmul()
