import boto3
import itertools
import numpy as np
from .matrix import BigSymmetricMatrix, BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name_uop, constant_zeros
from .matrix_init import local_numpy_init
import concurrent.futures as fs
import math
import os
import pywren
from pywren.executor import Executor
from scipy.linalg import cholesky, solve
import time

# this one is hard
def reshard(pwex, X, new_shard_sizes, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# These have some dependencies
def _argmin_remote(X, block_idxs):
    X_block = X.get_block(*block_idxs)
    offset = block_idxs[0]*X.shard_sizes[0]
    return (block_idxs[1], offset + np.argmin(X_block, axis=0), np.min(X_block, axis=0))

def argmin(pwex, X, out_bucket=None, tasks_per_job=1):
    futures = pwex.map(lambda x: _argmin_remote(x, X), X.block_idxs)
    pywren.wait(futures)
    results = [f.result() for f in futures]
    if (axis == None):
        groups = [(None, results)]
    else:
        groups = itertools.groupby(sorted(results, key=itemgetter(axis)), key=itemgetter(0))
    results = []
    for _, group in groups:
        group = list(group)
        argmins = np.concatenate([g[1] for g in group], axis=axis)
        argminmin = np.argmin(np.vstack([g[2] for g in group]), axis=axis)
        results.append(argmins[argminmin, :])
    return np.hstack(results)


def argmax(pwex, X, out_bucket=None, tasks_per_job=1):
    mins = []
    for _, group in itertools.groupby(sorted(results, key=itemgetter(0)), key=itemgetter(0)):
        group = list(group)
        argmins = np.vstack([g[1] for g in group])
        argminmin = np.argmin(np.vstack([g[2] for g in group]), axis=0)
        mins.append(argmins[argminmin, np.arange(argmins.shape[1])])
    return np.hstack(mins)

def min(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def max(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def norm(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sum(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def prod(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# these have no dependencies
def abs(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def neg(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def square(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sqrt(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sin(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def cos(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def tan(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def exp(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def sign(pwex, X, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def elemwise_uop_func(pwex, X, f, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def power(pwex, X, k, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def block_matmul_update(L, X, L_bb_inv, block_0_idx, block_1_idx):
    #print("BLOCK_MATMUL_UPDATE", block_0_idx, block_1_idx)
    #print("BLOCK_MATMUL_UPDATE PREUPDATE\n", L.numpy())
    #print("X\n",X.numpy())
    #print("X_blocks_exist", X.block_idxs_exist)
    L_bb_inv = L_bb_inv.get_block(0,0)
    X_block = X.get_block(block_1_idx, block_0_idx).T
    L_block = X_block.T.dot(L_bb_inv)
    #print("X_block", X_block)
    #print("L_bb_inv", L_bb_inv)
    L.put_block(L_block, block_1_idx, block_0_idx)
    #print("BLOCK_MATMUL_UPDATE POSTUPDATE\n", L.numpy())
    return 0

def syrk_update(L, X, block_0_idx, block_1_idx, block_2_idx):
    #print("SYRK")
    #print("SYRK_PREUPDATE\n",L.numpy())
    #print("X\n",X.numpy())
    #print("X_blocks_exist", X.block_idxs_exist)
    #print(block_0_idx,block_1_idx,block_2_idx)
    block_1 = L.get_block(block_1_idx, block_0_idx)
    block_2 = L.get_block(block_2_idx, block_0_idx)

    old_block = X.get_block(block_2_idx, block_1_idx)
    X_block = X.get_block(block_1_idx, block_2_idx).T
    #print("old_block",old_block)
    #print("BLOCK_1",block_1)
    #print("BLOCK_2",block_2)
    update = old_block - block_2.dot(block_1.T)
    #print("update", update)
    L.put_block(update, block_2_idx, block_1_idx)
    #print("L21", L.get_block(1,0))
    #print("SYRK_POSTUPDATE\n",L.numpy())
    return 0

def chol(pwex, X, out_bucket=None, tasks_per_job=1):
    if (out_bucket == None):
        out_bucket = X.bucket
    out_key = generate_key_name_uop(X, "chol")
    L = BigMatrix(out_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]], parent_fn=constant_zeros)
    L.free()
    all_blocks = list(L.block_idxs)
    for j in X._block_idxs(0):
        if (j == 0):
            diag_block = X.get_block(j,j)
            A = X
        else:
            diag_block = L.get_block(j,j)
            A = L
        print("Iteration {0}, cholesky".format(j))
        L_bb = cholesky(diag_block.T)
        #print("L_bb", L_bb)
        L.put_block(L_bb.T, j, j)
        L_bb_inv = solve(L_bb, np.eye(L_bb.shape[0]))
        #print("L_bb_inv", L_bb)
        L_bb_inv_bigm = local_numpy_init(L_bb_inv, L_bb_inv.shape)
        #print("L_bb_inv", L_bb_inv_bigm.get_block(0,0))
        def pywren_run(x):
            return block_matmul_update(L, A, L_bb_inv_bigm, *x)
        column_blocks = [block for block in all_blocks if (block[0] == j and block[1] > j)]
        print("Iteration {0}, column update".format(j))
        t = time.time()
        futures = pwex.map(pywren_run, column_blocks)
        pywren.wait(futures)
        [f.result() for f in futures]
        print("Iteration {0} Column update took {1}".format(j, time.time() - t))
        #print("COLUMN_BLOCKS",column_blocks)
        def pywren_run_2(x):
            print(x)
            return syrk_update(L, A, j, *x)
        other_blocks = list([block for block in all_blocks if (block[0] > j and block[1] > j and block[0] <= block[1])])
        #print("OTHER_BLOCKS",other_blocks)
        t = time.time()
        print("Iteration {0}, trailing matrix update".format(j))
        futures = pwex.map(pywren_run_2, other_blocks)
        pywren.wait(futures)
        [f.result() for f in futures]
        print("Iteration {0} trailing matrix update took {1}".format(j, time.time() - t))
        #print("BLOCKS OF L EXIST ", L.block_idxs_exist)
        L_bb_inv_bigm.free()
    return L


