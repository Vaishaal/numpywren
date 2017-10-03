import boto3
import itertools
import numpy as np
from .matrix import BigSymmetricMatrix, BigMatrix
from .matrix_utils import load_mmap, chunk, generate_key_name
import concurrent.futures as fs
import math
import os
import pywren
from pywren.executor import Executor
import time


def _gemm_remote_0(block_pairs, XY, X, Y, reduce_idxs=[0]):
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XY_block = None
        for r in reduce_idxs:
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(r, bidx_1)
            if (XY_block is None):
                XY_block = block1.dot(block2)
            else:
                XY_block += block1.dot(block2)
        XY.put_block(XY_block, bidx_0, bidx_1)

def gemm(pwex, X, Y, out_bucket, tasks_per_job=1, local=False):

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
    reduce_idxs = Y._block_idxs(axis=1)
    print(reduce_idxs)

    root_key = generate_key_name(X, Y, "gemm")

    if (X.key == Y.key and (X.transposed ^ Y.transposed)):
        XY = BigSymmetricMatrix(root_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]])
    else:
        XY = BigMatrix(root_key, shape=(X.shape[0], Y.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], Y.shard_sizes[0]])

    num_out_blocks = len(XY.blocks)
    num_jobs = int(num_out_blocks/float(tasks_per_job))

    print("Total number of output blocks", len(XY.block_idxs))
    print("Total number of output blocks that exist", len(XY.blocks_exist))

    block_idxs_to_map = list(set(XY.block_idxs))

    print("Number of output blocks to generate ", len(block_idxs_to_map))

    chunked_blocks = list(chunk(list(chunk(block_idxs_to_map, tasks_per_job)), num_jobs))


    def pywren_run(x):
        return _gemm_remote_0(x, XY, X, Y, reduce_idxs=reduce_idxs)

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




