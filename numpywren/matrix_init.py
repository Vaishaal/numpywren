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
from .matrix_utils import generate_key_name_local_matrix

def local_numpy_init(X_local, shard_sizes, n_jobs=1, symmetric=False):
    print("Sharding matrix..... of shape {0}".format(X_local.shape))
    key = generate_key_name_local_matrix(X_local)
    if (not symmetric):
        bigm = BigMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes)
    else:
        bigm = BigSymmetricMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes)
    return shard_matrix(bigm, X_local, n_jobs=n_jobs)

def shard_matrix(bigm, X_local, n_jobs=1):
    all_bidxs = bigm.block_idxs
    all_blocks = bigm.blocks
    executor = fs.ThreadPoolExecutor(n_jobs)
    futures = []
    for (bidxs,blocks) in zip(all_bidxs, all_blocks):
        slices = [slice(s,e) for s,e in blocks]
        X_block = X_local.__getitem__(slices)
        future = executor.submit(bigm.put_block, X_block, *bidxs)
        futures.append(future)
        fs.wait(futures)
    [f.result() for f in futures]
    return bigm




