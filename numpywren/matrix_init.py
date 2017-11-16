import concurrent.futures as fs
import io
import itertools
import os
import time

import boto3
import cloudpickle
import numpy as np
import hashlib
from .matrix import BigMatrix, BigSymmetricMatrix
from .matrix_utils import generate_key_name_local_matrix, constant_zeros, MmapArray
from . import matrix_utils
import numpy as np


def local_numpy_init(X_local, shard_sizes, n_jobs=1, symmetric=False, exists=False, executor=None):
    #print("Sharding matrix..... of shape {0}".format(X_local.shape))
    print("Generating key name...")
    key = generate_key_name_local_matrix(X_local)
    if (not symmetric):
        bigm = BigMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes, dtype=X_local.dtype)
    else:
        bigm = BigSymmetricMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes, dtype=X_local.dtype)
    if (not exists):
        return shard_matrix(bigm, X_local, n_jobs=n_jobs, executor=executor)
    else:
        return bigm

def empty_result_matrix(X_sharded, function, shape=None, shard_sizes=None, symmetric=False, dtype=None):
    if (dtype == None):
        dtype = X_sharded.dtype
    if (shape == None):
        shape = X_sharded.shape
    if (shard_sizes == None):
        shard_sizes = X_sharded.shard_sizes
    #print("Sharding matrix..... of shape {0}".format(X_local.shape))
    key_hash = X_sharded.key
    function_hash = matrix_utils.hash_function(function)
    key = matrix_utils.hash_bytes(function_hash + key_hash)
    if (not symmetric):
        bigm = BigMatrix(key, shape=shape, shard_sizes=shard_sizes, dtype=dtype)
    else:
        bigm = BigSymmetricMatrix(key, shape=shape, shard_sizes=shard_sizes, dtype=dtype)
    return bigm

def mmap_put_block(bigm, mmap_array, bidxs_blocks):
    bidxs,blocks = zip(*bidxs_blocks)
    slices = [slice(s,e) for s,e in blocks]
    X_local = mmap_array.load()
    X_block = X_local.__getitem__(slices)
    #print("Uploading to {0}".format(bigm))
    #print("Uploading {0}".format(X_block.shape))
    #print("Uploading {0}".format(X_block))
    return bigm.put_block(X_block, *bidxs)

def _shard_matrix(bigm, X_local, n_jobs=1, executor=None):
    all_bidxs = bigm.block_idxs
    all_blocks = bigm.blocks
    executor = fs.ProcessPoolExecutor(n_jobs)
    futures = []
    for (bidxs,blocks) in zip(all_bidxs, all_blocks):
        slices = [slice(s,e) for s,e in blocks]
        X_block = X_local.__getitem__(slices)
        future = executor.submit(bigm.put_block, X_block, *bidxs)
        futures.append(future)
        fs.wait(futures)
    [f.result() for f in futures]
    return bigm


def shard_matrix(bigm, X_local, n_jobs=1, executor=None):
    print("SHARDING")
    all_bidxs = bigm.block_idxs
    all_blocks = bigm.blocks
    if (executor == None):
        executor = fs.ThreadPoolExecutor(n_jobs)
    print(executor)
    futures = []
    t = time.time()
    X_local_mmaped = np.memmap("/dev/shm/{0}".format(bigm.key), dtype=bigm.dtype, shape=bigm.shape, mode="w+")
    e = time.time()
    np.copyto(X_local_mmaped, X_local)
    X_local_mmap = MmapArray(X_local_mmaped, "r")
    for (bidxs,blocks) in zip(all_bidxs, all_blocks):
        slices = [slice(s,e) for s,e in blocks]
        X_block = X_local.__getitem__(slices)
        future = executor.submit(mmap_put_block, bigm, X_local_mmap, zip(bidxs, blocks))
        futures.append(future)
        fs.wait(futures)
    [f.result() for f in futures]
    return bigm



