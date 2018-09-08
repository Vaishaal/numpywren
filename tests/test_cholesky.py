from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner, frontend
from numpywren import compiler
from numpywren.matrix_utils import constant_zeros
from numpywren.matrix_init import shard_matrix
import dill
import numpywren as npw

import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import os
import boto3

class CholeskyTest(unittest.TestCase):
    def test_cholesky_single(self):
        X = np.random.randn(64, 64)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        shard_size = 16
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions, trailing, L_sharded = _chol(A_sharded)
        executor = pywren.lambda_executor
        config = npw.config.default()
        program = lp.LambdaPackProgram(instructions, config=config)
        program.start()
        executor = fs.ProcessPoolExecutor(1)
        print("starting program")
        future = executor.submit(job_runner.lambdapack_run, program)
        program.wait()
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        assert(np.allclose(L_npw, L))
        return

    def test_cholesky_multi(self):
        X = np.random.randn(64, 64)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        shard_size = 16
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions, trailing, L_sharded = _chol(A_sharded)
        executor = pywren.lambda_executor
        config = npw.config.default()
        program = lp.LambdaPackProgram(instructions, config=config)
        program.start()
        executor = fs.ProcessPoolExecutor(1)
        print("starting program")
        future = executor.submit(job_runner.lambdapack_run, program)
        program.wait()
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        assert(np.allclose(L_npw, L))
        return



