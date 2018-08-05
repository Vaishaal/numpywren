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
import boto3




def reduce_sum(A:BigMatrix, B:BigMatrix, N:int, b_fac:int):
    with reducer(expr=A[0, i],  var=i, start=0, end=N, b_fac=b_fac) as r:
        B[r.level, i] = add(*r.reduce_args)
        r.reduce_next(B[r.level, i])


class ReductionTest(unittest.TestCase):
    def test_reduction_one_step(self):
        np.random.seed(1)
        size = 256
        shard_size = 256
        np.random.seed(1)
        X = np.random.randn(shard_size,  size)
        shard_sizes = (shard_size, shard_size)
        X_sharded= BigMatrix("reduction_test_A", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        b_fac = 2
        #b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(1))/np.log2(b_fac))), 1)
        print("num_tree_levels", num_tree_levels)
        B_sharded= BigMatrix("reduction_test_B", shape=(num_tree_levels*shard_size, X_sharded.shape[1]), shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        r_sum = frontend.lpcompile(reduce_sum)
        config = npw.config.default()
        program_compiled = r_sum(X_sharded, B_sharded, X_sharded.num_blocks(1), b_fac)
        terminators = program_compiled.find_terminators()
        print("Terminators ", terminators)
        starters = program_compiled.starters
        starters_children = program_compiled.get_children(*starters[0])
        program_executable = lp.LambdaPackProgram(program_compiled, config=config)
        program_executable.start()
        num_cores = 1
        all_futures  = []
        executor = fs.ProcessPoolExecutor(num_cores)
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, program_executable, pipeline_width=1, idle_timeout=5, timeout=10))
        program_executable.wait()
        time.sleep(5)
        program_executable.free()

    def test_reduction_single_stage(self):
        np.random.seed(1)
        size = 256
        shard_size = 32
        np.random.seed(1)
        X = np.random.randn(32,  size)
        shard_sizes = (shard_size, shard_size)
        X_sharded= BigMatrix("reduction_test_A", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        b_fac = X_sharded.num_blocks(1)
        #b_fac = 2
        num_tree_levels = int(np.ceil(np.log2(X_sharded.num_blocks(1))/np.log2(b_fac)))
        B_sharded= BigMatrix("reduction_test_B", shape=(num_tree_levels*shard_size, X_sharded.shape[1]), shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        r_sum = frontend.lpcompile(reduce_sum)
        config = npw.config.default()
        program_compiled = r_sum(X_sharded, B_sharded, X_sharded.num_blocks(1), b_fac)
        starters = program_compiled.starters
        starters_children = program_compiled.get_children(*starters[0])
        program_executable = lp.LambdaPackProgram(program_compiled, config=config)
        program_executable.start()
        num_cores = 1
        all_futures  = []
        executor = fs.ProcessPoolExecutor(num_cores)
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, program_executable, pipeline_width=1, idle_timeout=5, timeout=10))
        program_executable.wait()
        time.sleep(5)
        program_executable.free()
        reduce_output = B_sharded.get_block(num_tree_levels - 1, 0)
        reduce_output_check = kernels.add(*[X[:, 32*(i):32*(i+1)] for i in range(X_sharded.num_blocks(1))])
        assert(np.allclose(reduce_output, reduce_output_check))

    def test_reduction_multi_stage(self):
        np.random.seed(1)
        size = 256
        shard_size = 32
        np.random.seed(1)
        X = np.random.randn(32,  size)
        shard_sizes = (shard_size, shard_size)
        X_sharded= BigMatrix("reduction_test_A", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        b_fac = 2
        num_tree_levels = int(np.ceil(np.log2(X_sharded.num_blocks(1))/np.log2(b_fac)))
        B_sharded= BigMatrix("reduction_test_B", shape=(num_tree_levels*shard_size, X_sharded.shape[1]), shard_sizes=shard_sizes, write_header=True)
        X_sharded.free()
        shard_matrix(X_sharded, X)
        r_sum = frontend.lpcompile(reduce_sum)
        config = npw.config.default()
        program_compiled = r_sum(X_sharded, B_sharded, X_sharded.num_blocks(1), b_fac)
        starters = program_compiled.starters
        starters_children = program_compiled.get_children(*starters[0])
        starters_children = program_compiled.get_children(*starters[1])
        program_executable = lp.LambdaPackProgram(program_compiled, config=config)
        program_executable.start()
        all_futures  = []
        num_cores = 16
        executor = fs.ProcessPoolExecutor(num_cores)
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, program_executable, pipeline_width=1, idle_timeout=5, timeout=10))
        program_executable.wait()
        time.sleep(5)
        program_executable.free()
        reduce_output = B_sharded.get_block(num_tree_levels - 1, 0)
        reduce_output_check = kernels.add(*[X[:, 32*(i):32*(i+1)] for i in range(X_sharded.num_blocks(1))])
        assert(np.allclose(reduce_output, reduce_output_check))












