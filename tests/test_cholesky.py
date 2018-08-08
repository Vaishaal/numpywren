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

def cholesky(O:BigMatrix, I:BigMatrix, S:BigMatrix,  N:int, truncate:int):
    # handle first loop differently
    O[0,0] = chol(I[0,0])
    for j in range(1,N - truncate):
        O[j,0] = trsm(O[0,0], I[j,0])
        for k in range(1,j+1):
            S[1,j,k] = syrk(I[j,k], O[j,0], O[k,0])

    for i in range(1,N - truncate):
        O[i,i] = chol(S[i,i,i])
        for j in range(i+1,N - truncate):
            O[j,i] = trsm(O[i,i], S[i,j,i])
            for k in range(i+1,j+1):
                S[i+1,j,k] = syrk(S[i,j,k], O[j,i], O[k,i])

def _chol(X, out_bucket=None, truncate=0):
    S = BigMatrix("Cholesky.Intermediate({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
    O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.shape[0], X.shape[0]), shard_sizes=(X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
    O.parent_fn = dill.dumps(constant_zeros)
    block_len = len(X._block_idxs(0))
    print("Truncate is ", truncate)
    program = frontend.lpcompile(cholesky)(O,X,S,int(np.ceil(X.shape[0]/X.shard_sizes[0])), truncate)
    print(program)
    s = program.find_starters()
    children = program.get_children(*s[0])
    print("Starters: " + str(s))
    print("Children: " + str(children))
    print("Grand Children: " + str(program.get_children(*children[0])))
    print("Terminators: " + str(program.find_terminators()))
    operator_expr = program.get_expr(s[0][0])
    inst_block = operator_expr.eval_operator(s[0][1])
    return program, S, O


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



