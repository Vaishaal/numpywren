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


def TSQR(A:BigMatrix, Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, N:int, bfac:int):
    with reducer(expr=A[j, 0], var=j, start=0, end=N, b_fac=bfac) as r:
        Vs[r.level, j], Ts[r.level, j], Rs[r.level, j] = qr_factor(*r.reduce_args)
        r.reduce_next(Rs[r.level, j])


class TSQRTest(unittest.TestCase):
    def test_tsqr_multi_stage_static(self):
        np.random.seed(1)
        size = 256
        shard_size = 16
        np.random.seed(1)
        np.random.seed(0)
        X = np.random.randn(size, size)
        Q,R = np.linalg.qr(X)
        v,t,r = kernels.fast_qr(X)

        shard_sizes = (shard_size, size)
        X_sharded= BigMatrix("tsqr_test_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        R_sharded= BigMatrix("tsqr_test_R", shape=(num_tree_levels*shard_size, X_sharded.shape[0]), shard_sizes=shard_sizes, write_header=True, safe=False)
        V_sharded= BigMatrix("tsqr_test_V", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)
        T_sharded= BigMatrix("tsqr_test_T", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)

        X_sharded.free()
        shard_matrix(X_sharded, X)
        X_sharded.get_block(0,0)
        tsqr = frontend.lpcompile(TSQR)
        config = npw.config.default()
        N_blocks = X_sharded.num_blocks(0)
        program_compiled = tsqr(X_sharded, V_sharded, T_sharded, R_sharded, N_blocks, b_fac)
        starters = program_compiled.starters
        terminators = program_compiled.find_terminators()
        starters_children = [y for x in starters for y in program_compiled.get_children(*x) if program_compiled.get_expr(y[0]) != program_compiled.return_expr]
        starters_children_parents = [y for x in starters_children for y in program_compiled.get_parents(*x) if program_compiled.get_expr(y[0]) != program_compiled.return_expr]
        starters_children = set([(y[0], tuple(y[1].items())) for y in starters_children])
        starters = set([(y[0], tuple(y[1].items())) for y in starters])
        starters_children_parents  = set([(y[0], tuple(y[1].items())) for y in starters_children_parents])
        print("Starters ", starters)
        print("Starters children", starters_children)
        print("Starters children parents", starters_children_parents)
        assert(len(starters) == int(np.ceil((size/shard_size)/b_fac)))
        assert(len(starters_children) == len(starters)/b_fac)
        assert(starters_children_parents == starters)


    def test_tsqr_fat_runtime(self):
        np.random.seed(1)
        size = 256
        shard_size = 32
        np.random.seed(0)
        X = np.random.randn(size, 16)
        Q,R = np.linalg.qr(X)
        v,t,r = kernels.fast_qr(X)
        Q0 = np.eye(v.shape[0]) - v.dot(t).dot(v.T)
        shard_sizes = (shard_size, X.shape[1])
        X_sharded= BigMatrix("tsqr_test_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        R_sharded= BigMatrix("tsqr_test_R", shape=(num_tree_levels*shard_size, X_sharded.shape[0]), shard_sizes=shard_sizes, write_header=True, safe=False)
        T_sharded= BigMatrix("tsqr_test_T", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)
        V_sharded= BigMatrix("tsqr_test_V", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)

        X_sharded.free()
        shard_matrix(X_sharded, X)
        X_sharded.get_block(0,0)
        tsqr = frontend.lpcompile(TSQR)
        config = npw.config.default()
        N_blocks = X_sharded.num_blocks(0)
        program_compiled = tsqr(X_sharded, V_sharded, T_sharded, R_sharded, N_blocks, b_fac)
        starters = program_compiled.starters
        terminators = program_compiled.find_terminators()
        program_executable = lp.LambdaPackProgram(program_compiled, config=config)
        print(program_compiled)
        program_executable.start()
        num_cores = 1
        all_futures  = []
        job_runner.lambdapack_run(program_executable, pipeline_width=1, idle_timeout=5, timeout=60)
        executor = fs.ThreadPoolExecutor(num_cores)
        for i in range(num_cores):
            all_futures.append(executor.submit(job_runner.lambdapack_run, program_executable, pipeline_width=1, idle_timeout=5, timeout=60))
        program_executable.wait()
        time.sleep(5)
        program_executable.free()
        R_remote = R_sharded.get_block(num_tree_levels - 1, 0)
        sign_matrix_local = np.eye(R.shape[0])
        sign_matrix_remote = np.eye(R.shape[0])
        sign_matrix_local[np.where(np.diag(R) <= 0)]  *= -1
        sign_matrix_remote[np.where(np.diag(R_remote) <= 0)]  *= -1
        # make the signs match
        R_remote *= sign_matrix_remote
        R  *= sign_matrix_local
        print("DIFF\n", np.max(np.abs(R - R_remote)))
        print("DIFF IDX \n", np.argmax(np.abs(R - R_remote)))
        print("DIFF\n", np.max(np.abs(np.abs(R) - np.abs(R_remote))))
        assert(np.allclose(R, R_remote))
        executor.shutdown(wait=False)
        return

