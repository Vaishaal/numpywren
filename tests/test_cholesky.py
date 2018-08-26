from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner
from numpywren import compiler
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
import boto3


class CholeskyTest(unittest.TestCase):
    def test_cholesky_single(self):
        X = np.random.randn(3,3)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=A.shape, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions, trailing, L_sharded = compiler._chol(A_sharded)
        executor = pywren.lambda_executor
        pywren_config = pwex.config
        config = npw.config.default()
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=pywren_config, config=config)
        program.start()
        job_runner.lambdapack_run(program)
        program.wait()
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        assert(np.allclose(L_npw, L))


    def test_cholesky_multi(self):
        np.random.seed(1)
        size = 128
        shard_size = 32
        np.random.seed(1)
        X = np.random.randn(size, 128)
        A = X.dot(X.T) + np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions, trailing, L_sharded = compiler._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.lambda_executor
        config = npw.config.default()
        pywren_config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=pywren_config, config=config)
        program.start()
        #job_runner.main(program, program.queue_url)
        job_runner.lambdapack_run(program)
        print("Program status")
        print(program.program_status())
        program.free()
        print(L_sharded.shape)
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        print(L_npw)
        print(L)
        assert(np.allclose(L_npw, L))

    def test_cholesky_lambda_single(self): 
        np.random.seed(1)
        size = 128
        shard_size = 128
        num_cores = 1
        np.random.seed(1)
        X = np.random.randn(size, 128)
        A = X.dot(X.T) + np.eye(X.shape[0])
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes, write_header=True)
        A_sharded.free()
        shard_matrix(A_sharded, A)
        instructions, trailing, L_sharded = compiler._chol(A_sharded)
        pwex = pywren.default_executor()
        executor = pywren.standalone_executor
        config = npw.config.default()
        pywren_config = pwex.config
        program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=pywren_config, config=config)
        print(program)
        program.start()
        num_cores = 1
        futures = pwex.map(lambda x: job_runner.lambdapack_run(program, timeout=10), range(num_cores), exclude_modules=["site-packages"])
        #futures = pwex.map(lambda x: x, range(num_cores), exclude_modules=["site-packages"], extra_env=redis_env)
        print("waiting for result")
        futures[0].result()
        pywren.wait(futures)
        [f.result() for f in futures]
        program.free()
        L_npw = L_sharded.numpy()
        L = np.linalg.cholesky(A)
        assert(np.allclose(L_npw, L))



