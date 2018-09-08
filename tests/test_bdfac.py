from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner, frontend
from numpywren import compiler
from numpywren.matrix_utils import constant_zeros
from numpywren.matrix_init import shard_matrix
from numpywren import kernels
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


def BDFAC(VLs:BigMatrix, TLs:BigMatrix, Rs:BigMatrix, Sigma:BigMatrix, VRs:BigMatrix, TRs:BigMatrix, S0:BigMatrix, S1:BigMatrix, N:int, truncate:int):
    b_fac = 2
    for i in range(0, N):
        N_tree = ceiling(log(N - i)/log(2))
        for j in range(i, N):
            VLs[j, i, N_tree], TLs[j, i, N_tree], Rs[j, i, N_tree] = qr_factor(S1[j, i, i, 0])

        with reducer(expr=Rs[j, i, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
            VLs[j, i, N_tree - r.level - 1], TLs[j, i, N_tree - r.level - 1], Rs[j, i, N_tree - r.level - 1] = qr_factor(*r.reduce_args)
            r.reduce_next(Rs[j, i, N_tree - r.level - 1])

        # flat trailing matrix update
        for j in range(i, N):
            for k in range(i+1, N):
                S0[j, k, i+1, N_tree] = qr_leaf(VLs[j, i, N_tree], TLs[j, i, N_tree], S1[j, k, i, 0])

        for k in range(i+1, N):
            with reducer(expr=S0[j, k, i+1, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
                S0[j, k, i+1, N_tree - r.level - 1], S0[j + b_fac**r.level, k, i+1, 0]  = qr_trailing_update(VLs[j, i, N_tree - r.level - 1], TLs[j, i, N_tree - r.level - 1], *r.reduce_args)
                r.reduce_next(S0[j, k, i+1, N_tree - r.level - 1])

        for k in range(i+1, N):
            Rs[i, k, 0]  = identity(S0[i, k, i+1, 0])

        for j in range(i, N):
            VRs[j, i, N_tree], TRs[j, i, N_tree], Sigma[j, i, N_tree] = lq_factor(S0[j, i, i, 0])

        with reducer(expr=Sigma[j, i, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
            VRs[j, i, N_tree - r.level - 1], TRs[j, i, N_tree - r.level - 1], Sigma[j, i, N_tree - r.level - 1] = lq_factor(*r.reduce_args)
            r.reduce_next(Sigma[j, i, N_tree - r.level - 1])

        for j in range(i, N):
            for k in range(i+1, N):
                S1[j, k, i+1, N_tree] = lq_leaf(VRs[j, i, N_tree], TRs[j, i, N_tree], S0[j, k, i, 0])

        for k in range(i+1, N):
            with reducer(expr=S1[j, k, i+1, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
                S1[j, k, i+1, N_tree - r.level - 1], S1[j + b_fac**r.level, k, i+1, 0]  = lq_trailing_update(VRs[j, i, N_tree - r.level - 1], TRs[j, i, N_tree - r.level - 1], *r.reduce_args)
                r.reduce_next(S1[j, k, i+1, N_tree - r.level - 1])

        for k in range(i+1, N):
            Sigma[i, k, 0]  = identity(S1[i, k, i+1, 0])








class BDFACTest(unittest.TestCase):
    def test_bdfac(self):
        N = 8
        shard_size = 2
        shard_sizes = (shard_size, shard_size)
        X = np.random.randn(8, 8)
        X_sharded = BigMatrix("BDFAC_input_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        shard_matrix(X_sharded, X)
        N_blocks = X_sharded.num_blocks(0)
        b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        async def parent_fn(self, loop, *block_idxs):
            if (block_idxs[-1] == 0 and block_idxs[-2] == 0):
                return await X_sharded.get_block_async(None, *block_idxs[:-2])
        VLs = BigMatrix("VL", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        TLs = BigMatrix("TL", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        VRs = BigMatrix("VR", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        TRs = BigMatrix("TR", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        Rs = BigMatrix("R", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        S0 = BigMatrix("S0", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True, parent_fn=parent_fn, safe=False)
        S1 = BigMatrix("S1", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True, parent_fn=parent_fn, safe=False)
        Sigma = BigMatrix("Sigma", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False, parent_fn=parent_fn)
        pc = frontend.lpcompile(BDFAC)(VLs, TLs, Rs, Sigma, VRs, TRs, S0, S1, N_blocks, 0)

if __name__ == "__main__":
    test  = BDFACTest()
    test.test_bdfac()

