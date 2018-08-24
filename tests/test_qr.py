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


def QR(Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, S:BigMatrix, N:int, truncate:int):
    b_fac = 2
    for i in range(0, N):
        N_tree = ceiling(log(N - i)/log(2))
        for j in range(i, N):
            Vs[j, i, N_tree], Ts[j, i, N_tree], Rs[j, i, N_tree] = qr_factor(S[j, i, i, 0])

        with reducer(expr=Rs[j, i, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
            Vs[j, i, N_tree - r.level - 1], Ts[j, i, N_tree - r.level - 1], Rs[j, i, N_tree - r.level - 1] = qr_factor(*r.reduce_args)
            r.reduce_next(Rs[j, i, N_tree - r.level - 1])

        # flat trailing matrix update
        for j in range(i, N):
            for k in range(i+1, N):
                S[j, k, i+1, N_tree] = qr_leaf(Vs[j, i, N_tree], Ts[j, i, N_tree], S[j, k, i, 0])

        for k in range(i+1, N):
            with reducer(expr=S[j, k, i+1, N_tree], var=j, start=i, end=N, b_fac=b_fac) as r:
                S[j, k, i+1, N_tree - r.level - 1], S[j + b_fac**r.level, k, i+1, 0]  = qr_trailing_update(Vs[j, i, N_tree - r.level - 1], Ts[j, i, N_tree - r.level - 1], *r.reduce_args)
                r.reduce_next(S[j, k, i+1, N_tree - r.level - 1])

        for k in range(i+1, N):
            Rs[i, k, 0]  = identity(S[i, k, i+1, 0])


class QRTest(unittest.TestCase):
    def test_qr_single_static(self):
        X = np.random.randn(64, 64)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        N = 128
        shard_size = 32
        shard_sizes = (shard_size, shard_size)
        X =  np.random.randn(N, N)
        X_sharded = BigMatrix("QR_input_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        #shard_matrix(X_sharded, X)
        N_blocks = X_sharded.num_blocks(0)
        b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        Vs = BigMatrix("Vs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True)
        Ts = BigMatrix("Ts", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True)
        Rs = BigMatrix("Rs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True)

        Ss = BigMatrix("Ss", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True)

        pc = frontend.lpcompile(QR)(Vs, Ts, Rs, Ss, N_blocks, 0)
        s = pc.find_starters()
        print(s)
        print(pc.get_children(3, {'__LEVEL__': 1, '__nl3__': 0, 'i': 0, 'j': 0, 'k': 1}))
        print(pc.get_children(3, {'__LEVEL__': 0, '__nl3__': 0, 'i': 1, 'j': 1, 'k': 2}))
        print(pc.get_parents(0, {'i': 2, 'j': 2}))
        parents = pc.get_parents(0, {'i': 3, 'j': 3})
        print(parents)
        children = pc.get_children(*parents[0])
        print(children)
        assert((0, {'i': 3, 'j': 3}) in children)

    def test_qr_single_dynamic(self):
        X = np.random.randn(8, 8)
        pwex = pywren.default_executor()
        N = 8
        shard_size = 2
        shard_sizes = (shard_size, shard_size)
        np.random.seed(0)
        X =  np.random.randn(N, N)
        v0, t0, r0 = kernels.qr_factor(X[:4, :4])
        v1,t1, r1 = kernels.qr_factor(X[4:, :4])

        v00, t00, r_out = kernels.qr_factor(np.vstack((r0,r1)))
        #print(v0)
        #print(t0)
        #print(r_out)
        Q0 = (np.eye(v0.shape[0]) - v0.dot(t0).dot(v0.T))
        Q1 = (np.eye(v1.shape[0]) - v1.dot(t1).dot(v1.T))


        I = np.eye(v0.shape[0])
        Q = I - v0.dot(t0).dot(v0.T)

        s0 = kernels.qr_leaf(v0, t0, X[:4, 4:])
        s1 = kernels.qr_leaf(v1, t1, X[4:, 4:])
        #print("s0", s0)
        s01, s11 = kernels.qr_trailing_update(v00, t00, s0, s1)
        _, _, r11 = kernels.qr_factor(s11)
        #print("numpy", np.linalg.qr(X)[-1][4:, 4:])
        #print("lapack", r11)
        X_sharded = BigMatrix("QR_input_X", shape=X.shape, shard_sizes=shard_sizes, write_header=True)
        shard_matrix(X_sharded, X)
        #print("First block", X_sharded.get_block(0,0))
        #print("Second block", X_sharded.get_block(1,0))
        N_blocks = X_sharded.num_blocks(0)
        b_fac = 2
        num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
        async def parent_fn(self, loop, *block_idxs):
            if (block_idxs[-1] == 0 and block_idxs[-2] == 0):
                return await X_sharded.get_block_async(None, *block_idxs[:-2])
        Vs = BigMatrix("Vs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        Ts = BigMatrix("Ts", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        Rs = BigMatrix("Rs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
        Ss = BigMatrix("Ss", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True, parent_fn=parent_fn, safe=False)
        print("N BLOCKS", N_blocks)
        pc = frontend.lpcompile(QR)(Vs, Ts, Rs, Ss, N_blocks, 0)
        #print(pc.starters)
        #print(pc.get_children(*pc.starters[0]))
        #print(pc.num_terminators)
        #print(pc.find_terminators())
        #print(N_blocks)
        print("========="*25)
        #print(pc.get_children(1, {'i': 0, 'j': 0, '__LEVEL__': 0}))
        print(pc.get_parents(0, {'i': 1, 'j': 2}))
        print(pc.get_children(3, {'i': 0, 'k': 1, 'j': 0, '__LEVEL__': 0}))
        print(pc.get_children(3, {'i': 0, 'k': 1, 'j': 0, '__LEVEL__': 1}))
        return
        #print(pc.get_parents(1, {'i': 0, 'j': 0, '__LEVEL__': 1}))

        #print(pc.get_parents(2, {'i': 0, 'j': 1, 'k': 1}))
        #print(pc.get_parents(4, {'i': 0, 'k': 1}))
        #print(pc.get_children(3, {'i': 0, 'k': 1, 'j': 0, '__LEVEL__': 0}))
        #print(pc.get_children(3, {'i': 0, 'k': 1, 'j': 0, '__LEVEL__': 0}))
        #print(pc.get_children(0, {'i': 0, 'j': 0}))
        #print(pc.get_children(1, {'i': 0, 'j': 0, '__LEVEL__': 0}))
        #print(pc.get_children(2, {'i': 0, 'j': 0, 'k': 1}))
        #print(pc.get_children(2, {'i': 0, 'j': 1, 'k': 1}))
        all_nodes = pc.find_terminators()
        for node in all_nodes:
            print(f"CALLING GET parent for {node}")
            parents = pc.get_parents(*node)
            children = pc.get_children(*node)
            for child in children:
                if (pc.get_expr(child[0]) == pc.return_expr): continue
                parent_children = pc.get_parents(*child)
                assert node in parent_children
            for parent in parents:
                print(f"CALLING GET CHILDREN for {parent} who is parent of {node}")
                children_parents = pc.get_children(*parent)


                children_parents = [x for x in children_parents if pc.get_expr(x[0]) != pc.return_expr]
                print("parent", parent)
                print("children", children_parents)
                assert node in children_parents


        return
        config = npw.config.default()
        program = lp.LambdaPackProgram(pc, config=config)
        program.start()
        executor = fs.ProcessPoolExecutor(1)
        executor.submit(job_runner.lambdapack_run, program, pipeline_width=1, timeout=30, idle_timeout=30)
        program.wait()
        print(Rs.get_block(1, 1, 0))
        print(np.linalg.qr(X)[1][4:, 4:])
        return
        exit()

if __name__ == "__main__":
    test  = QRTest()
    test.test_qr_single_dynamic()

