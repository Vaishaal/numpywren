import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import numpy as np
import pytest
import pywren
import unittest
import os
import scipy.linalg
import warnings
from numpywren import frontend


def FORWARD_SUB(x:BigMatrix, L:BigMatrix, b:BigMatrix, S:BigMatrix, N:int, b_fac:int):
    for i in range(N):
        N_tree = ceiling(log(i+1)/log(b_fac))
        for j in range(0, i):
            S[i, j, N_tree] = gemm(L[i,j], x[j])

        with reducer(expr=S[i, j, N_tree], var=j, start=0, end=i, b_fac=b_fac) as r:
            S[i, j, N_tree - r.level - 1]  = add(*r.reduce_args)
            r.reduce_next(S[i, j, N_tree - r.level - 1])
        x[i] = trsm_sub(L[i,i] , S[i, j, 0], b[i])

class TriSolveTest(unittest.TestCase):
    def test_trisolve(self):
        N = 8
        L = np.tril(np.random.randn(N, N))
        b = np.random.randn(N)
        b_fac = 4
        shard_size = 2
        shard_sizes = (shard_size, shard_size)
        L_sharded = BigMatrix("F_solve_input_L", shape=L.shape, shard_sizes=shard_sizes, write_header=True)
        x_sharded = BigMatrix("x", shape=b.shape, shard_sizes=(shard_size,), write_header=True)
        num_tree_levels = max(int(np.ceil(np.log2(x_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        b_sharded = BigMatrix("b", shape=b.shape, shard_sizes=(shard_size,), write_header=True)
        S_sharded = BigMatrix("s", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), safe=False, write_header=True)
        shard_matrix(L_sharded, L)
        shard_matrix(b_sharded, b)
        N_blocks = x_sharded.num_blocks(0)
        num_tree_levels = max(int(np.ceil(np.log2(L_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        print("N_blocks")
        pc = frontend.lpcompile(FORWARD_SUB)(x_sharded, L_sharded, b_sharded, S_sharded, N_blocks, b_fac)
        print(pc.starters)
        print("=============")
        print(pc.get_children(1, {'i': 1, 'j': 0, '__LEVEL__': 0}))
        print(pc.get_children(1, {'i': 1, 'j': 0, '__LEVEL__': 1}))




