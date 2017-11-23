import boto3
import itertools
import numpy as np
from .matrix import BigSymmetricMatrix, BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name_binop, constant_zeros
from . import matrix_utils
from .matrix_init import local_numpy_init
import concurrent.futures as fs
import math
import os
import pywren
from pywren.executor import Executor
from scipy.linalg import cholesky, solve
import time


def _gemm_remote_0(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64):
    print(reduce_idxs)
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XY_block = None
        X.dtype = dtype
        Y.dtype = dtype
        for r in reduce_idxs:
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(r, bidx_1)
            if (XY_block is None):
                XY_block = block1.dot(block2)
            else:
                XY_block += block1.dot(block2)
        XY.put_block(XY_block, bidx_0, bidx_1)

def _gemm_remote_1(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64):
    os.system("sudo mount -o remount,size=50g /dev/shm")
    X.dtype = dtype
    Y.dtype = dtype
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        block0 = matrix_utils.get_row(X, bidx_0, mmap_loc="/dev/shm/block_0")
        block1 = matrix_utils.get_col(Y, bidx_1, mmap_loc="/dev/shm/block_1")
        XY_block = block0.dot(block1)
        XY.put_block(XY_block, bidx_0, bidx_1)


def gemm(pwex, X, Y, out_bucket=None, tasks_per_job=1, local=False, dtype=np.float64, overwrite=True):

    '''
        Compute XY return
        @param pwex - Execution context
        @param X - rhs matrix
        @param Y - lhs matrix
        @param tasks_per_job - number of tasks per job
        @param out_bucket - bucket job writes to
        @param num_jobs - how many lambdas to run
        @param local - run locally? #TODO remove once local pywren executor is provided
    '''
    # 0 -> 1 or 1 -> 0

    reduce_idxs = Y._block_idxs(axis=0)
    if (out_bucket == None):
        out_bucket = X.bucket

    root_key = generate_key_name_binop(X, Y, "gemm")
    if (Y.shard_sizes[0] !=  X.shard_sizes[1]):
        raise Exception("X dim 1 shard size must match Y dim 0 shard size")
    if (X.key == Y.key and (X.transposed ^ Y.transposed)):
        XY = BigSymmetricMatrix(root_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]], dtype=dtype)
    else:
        XY = BigMatrix(root_key, shape=(X.shape[0], Y.shape[1]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], Y.shard_sizes[1]], dtype=dtype)

    num_out_blocks = len(XY.blocks)
    if (tasks_per_job > num_out_blocks):
        tasks_per_job = 1
    num_jobs = int(num_out_blocks/float(tasks_per_job))

    print("Out Shape", XY.shape)
    print("Total number of output blocks", len(XY.block_idxs))
    print("Total number of output blocks that exist", len(XY.blocks_exist))

    if (overwrite):
        block_idxs_to_map = list(set(XY.block_idxs))
    else:
        block_idxs_to_map = list(set(XY.block_idxs_not_exist))

    print("Number of output blocks to generate ", len(block_idxs_to_map))
    chunked_blocks = list(chunk(list(chunk(block_idxs_to_map, tasks_per_job)), num_jobs))
    if (isinstance(pwex.invoker, pywren.queues.SQSInvoker)):
        def pywren_run(x):
            return _gemm_remote_1(x, XY, X, Y, reduce_idxs=reduce_idxs, dtype=dtype)
    else:
        def pywren_run(x):
            return _gemm_remote_0(x, XY, X, Y, reduce_idxs=reduce_idxs, dtype=dtype)

    all_futures = []
    for i, c in enumerate(chunked_blocks):
        print("Submitting job for chunk {0} in axis 0".format(i))
        if (local):
            list(map(pywren_run, c))
        else:
            s = time.time()
            futures = pwex.map(pywren_run, c)
            e = time.time()
            print("Pwex Map Time {0}".format(e - s))
            all_futures.append((i,futures))

    if (local):
        return XY

    for i, futures, in all_futures:
        pywren.wait(futures)
        [f.result() for f in futures]

    return XY

# matrix vector multiply
# hard
def gemv(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# symmetric rank k update
# hard
def syrk(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# very hard
def posv(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError




# easy
def add(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# easy
def sub(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# easy
def mul(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

# easy
def div(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def logical_and(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def logical_or(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def xor(pwex, X, Y, out_bucket=None, tasks_per_job=1):
    raise NotImplementedError

def elemwise_binop_func(pwex, X, Y, f, out_bucket=None, tasks_per_job=1, local=False):
    raise NotImplementedError

