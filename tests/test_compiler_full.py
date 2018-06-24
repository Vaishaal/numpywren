from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner
from numpywren import compiler
from numpywren.matrix_init import shard_matrix
import numpywren.wait
import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import os
import boto3
import dill

def cholesky(O:BigMatrix, I:BigMatrix, S:BigMatrix,  N:int, truncate:int):
    # handle first loop differently
    O[N,0,0] = cholesky(I[0,0,0])
    for j in range(1,N):
        O[N,j,0] = trsm(O[N,0,0], I[0,j,0])
        for k in range(1,j+1):
            S[1,j,k] = syrk(I[0,j,k], O[N,j,0], O[N,k,0])

    for i in range(1,N):
        O[N,i,i] = cholesky(S[i,i,i])
        for j in range(i+1,N):
            O[N,i,j] = trsm(O[N,i,i], S[i,i,j])
            for k in range(i+1,j+1):
                S[i+1,j,k] = syrk(S[i,j,k], O[N,j,i], O[N,k,i])

class CompilerTest(unittest.TestCase):
    def test_compiler_simple(self):
        N = 1281167
        truncate = 0
        X = BigMatrix("CholeskyInput", shape=(int(N),int(N)), shard_sizes=(4096, 4096), write_header=True)
        O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
        block_len = len(X._block_idxs(0))
        parent_fn = dill.dumps(compiler.make_3d_input_parent_fn_async(X))
        O.parent_fn = parent_fn
        program = compiler.lpcompile(cholesky, inputs=["I"], outputs=["O"])(O=O,I=O,S=O,N=int(np.ceil(X.shape[0]/X.shard_sizes[0])), truncate=truncate)
        assert len(program.find_starters()) == 1
        assert len(program.find_terminators()) == 49141

    def test_compiler_parent_children(self):
        N  = 1281167
        truncate = 0
        X = BigMatrix("CholeskyInput", shape=(int(N),int(N)), shard_sizes=(4096, 4096), write_header=True)
        O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), write_header=True)
        block_len = len(X._block_idxs(0))
        parent_fn = dill.dumps(compiler.make_3d_input_parent_fn_async(X))
        O.parent_fn = parent_fn
        s = time.time()
        program = compiler.lpcompile(cholesky, inputs=["I"], outputs=["O"])(O=O,I=O,S=O,N=int(np.ceil(X.shape[0]/X.shard_sizes[0])), truncate=truncate)
        e = time.time()
        print("COMPILE TIME", e - s)
        s = time.time()
        starter = program.find_starters()[0]
        e = time.time()
        print("STARTER TIME", e - s)
        depth = 2
        t = time.time()
        children = program.get_children(*starter)
        e  = time.time()
        for child in children:
            t = time.time()
            parents = program.get_parents(*child)
            e  = time.time()
            assert(parents[0] == starter)
        assert(len(children) == X.num_blocks(1) - 1)


