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

def f1_if(I:BigMatrix, O:BigMatrix, N:int):
    for i in range(N):
        if ((i % 2) == 0):
            O[i] = mul(1, I[i])
        else:
            O[i] = mul(2, I[i])

def f1_if_nested(I:BigMatrix, O:BigMatrix, N:int):
    for i in range(N):
        if ((i % 2) == 0):
            if ((i % 3) == 0):
                O[i] = mul(3, I[i])
            else:
                O[i] = mul(1, I[i])
        else:
            O[i] = mul(2, I[i])

def f1_if_or(I:BigMatrix, O:BigMatrix, N:int):
    for i in range(N):
        if ((i % 2) == 0 or (i%3) == 0):
            O[i] = mul(1, I[i])
        else:
            O[i] = mul(2, I[i])




class IfTest(unittest.TestCase):
    def test_if_static(self):
        X = np.random.randn(64, 64)
        shard_sizes = (int(X.shape[0]/8), X.shape[1])
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        assert(p.starters == p.find_terminators())
        for s, var_values in p.starters:
            if(var_values['i'] % 2 == 0):
                assert s == 0
            else:
                assert s == 1

    def test_if_static_nested(self):
        X = np.random.randn(64, 64)
        shard_sizes = (int(X.shape[0]/8), X.shape[1])
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if_nested)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        assert(p.starters == p.find_terminators())
        for s, var_values in p.starters:
            i = var_values['i']
            if(i % 2 == 0 and (not i % 3 == 0)):
                assert s == 1
            elif(i % 2 == 0 and (i % 3 == 0)):
                assert s == 0
            else:
                assert s == 2

    def test_if_static_or(self):
        X = np.random.randn(64, 64)
        shard_sizes = (int(X.shape[0]/8), X.shape[1])
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if_or)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        print(p.starters)
        assert(p.starters == p.find_terminators())
        for s, var_values in p.starters:
            i = var_values['i']
            if(i % 2 == 0 or (i % 3 == 0)):
                assert s == 0
            else:
                assert s == 1

    def test_nested_if_run(self):
        X = np.random.randn(64)
        shard_sizes = (int(X.shape[0]/8),)
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if_nested)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        num_cores = 1
        executor = fs.ProcessPoolExecutor(num_cores)
        config = npw.config.default()
        p_ex = lp.LambdaPackProgram(p, config=config)
        p_ex.start()
        all_futures = []
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, p_ex, pipeline_width=1, idle_timeout=5, timeout=60))
        p_ex.wait()
        time.sleep(5)
        p_ex.free()
        for i in range(X_sharded.num_blocks(0)):
            Ob = O_sharded.get_block(i)
            Xb = X_sharded.get_block(i)
            if ((i % 2) == 0 and ((i % 3) == 0)):
                assert(np.allclose(Ob, 3*Xb))
            elif ((i % 2) == 0):
                assert(np.allclose(Ob, Xb))
            else:
                assert(np.allclose(Ob, 2*Xb))

    def test_if_or_run(self):
        X = np.random.randn(64)
        shard_sizes = (int(X.shape[0]/8),)
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if_or)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        num_cores = 1
        executor = fs.ProcessPoolExecutor(num_cores)
        config = npw.config.default()
        p_ex = lp.LambdaPackProgram(p, config=config)
        p_ex.start()
        all_futures = []
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, p_ex, pipeline_width=1, idle_timeout=5, timeout=60))
        p_ex.wait()
        time.sleep(5)
        p_ex.free()
        for i in range(X_sharded.num_blocks(0)):
            Ob = O_sharded.get_block(i)
            Xb = X_sharded.get_block(i)
            if ((i % 2) == 0 or (i % 3) == 0):
                assert(np.allclose(Ob, 1*Xb))
            else:
                assert(np.allclose(Ob, 2*Xb))



    def test_if_run(self):
        X = np.random.randn(64)
        shard_sizes = (int(X.shape[0]/8),)
        X_sharded= BigMatrix("if_test", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        O_sharded= BigMatrix("if_test_output", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        f = frontend.lpcompile(f1_if)
        p  = f(X_sharded, O_sharded, X_sharded.num_blocks(0))
        num_cores = 1
        executor = fs.ProcessPoolExecutor(num_cores)
        config = npw.config.default()
        p_ex = lp.LambdaPackProgram(p, config=config)
        p_ex.start()
        all_futures = []
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, p_ex, pipeline_width=1, idle_timeout=5, timeout=60))
        p_ex.wait()
        time.sleep(5)
        p_ex.free()
        for i in range(X_sharded.num_blocks(0)):
            Ob = O_sharded.get_block(i)
            Xb = X_sharded.get_block(i)
            if ((i % 2) == 0):
                assert(np.allclose(Ob, 1*Xb))
            else:
                assert(np.allclose(Ob, 2*Xb))

