import concurrent.futures as fs
import io
import itertools
import os
import time

import boto3
import cloudpickle
import numpy as np
import hashlib
from .matrix import BigMatrix
from . import matrix
from .matrix_utils import generate_key_name_local_matrix, constant_zeros, MmapArray
from . import matrix_utils
import numpywren as npw
import numpy as np
import pywren


def local_numpy_init(X_local, shard_sizes, n_jobs=1, symmetric=False, exists=False, executor=None, write_header=False, bucket=matrix.DEFAULT_BUCKET, overwrite=True, **kwargs):
    key = generate_key_name_local_matrix(X_local)
    if (not symmetric):
        bigm = BigMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes, dtype=X_local.dtype, write_header=write_header, bucket=bucket, **kwargs)
    else:
        bigm = BigSymmetricMatrix(key, shape=X_local.shape, shard_sizes=shard_sizes, dtype=X_local.dtype, write_header=write_header, bucket=bucket, **kwargs)
    if (not exists):
        return shard_matrix(bigm, X_local, n_jobs=n_jobs, executor=executor, overwrite=overwrite)
    else:
        return bigm

def empty_result_matrix(X_sharded, function, args, shape=None, shard_sizes=None, symmetric=False, dtype=None, write_header=False):
    if (dtype == None):
        dtype = X_sharded.dtype
    if (shape == None):
        shape = X_sharded.shape
    if (shard_sizes == None):
        shard_sizes = X_sharded.shard_sizes
    #print("Sharding matrix..... of shape {0}".format(X_local.shape))
    key_hash = X_sharded.key
    function_hash = matrix_utils.hash_function(function)
    args_hash = matrix_utils.hash_args(args)
    key = matrix_utils.hash_string(function_hash + key_hash + args_hash)
    if (not symmetric):
        bigm = BigMatrix(key, shape=shape, shard_sizes=shard_sizes, dtype=dtype, write_header=write_header, bucket=X_sharded.bucket)
    else:
        bigm = BigSymmetricMatrix(key, shape=shape, shard_sizes=shard_sizes, dtype=dtype, write_header=write_header, bucket=X_sharded.bucket)
    return bigm

def mmap_put_block(bigm, mmap_array, bidxs_blocks):
    bidxs,blocks = zip(*bidxs_blocks)
    slices = [slice(s,e) for s,e in blocks]
    X_local = mmap_array.load()
    X_block = X_local.__getitem__(slices)
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


def shard_matrix(bigm, X_local, n_jobs=1, executor=None, overwrite=True):
    if (overwrite):
        all_bidxs = bigm.block_idxs
        all_blocks = bigm.blocks
    else:
        all_bidxs = bigm.block_idxs_not_exist
        all_blocks = bigm.blocks_not_exist

    if (executor == None):
        executor = fs.ThreadPoolExecutor(n_jobs)
    futures = []
    t = time.time()
    X_local_mmaped = np.memmap("{0}/{1}".format(npw.TMP_DIR, bigm.key), dtype=bigm.dtype, shape=bigm.shape, mode="w+")
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



def reshard_down(bigm, breakdowns, pwex=None):
    ''' Return a new bigm whose shard sizes are bigm.shard_sizes/break_downs
        if a pwex is provided reshards in parallel, else reshards locally (very slow)
        This will essentially break down a single block in bigm into several evenly sized sub blocks, breakdowns is a list of integers detailing how much to break a given dimension down into. breakdowns = [2,2] would break each dimension down into 2 independent block so a 4 x 4 block would be replaced by 4, 2 x 2 blocks.
    '''

    for x,y in zip(bigm.shard_sizes, breakdowns):
        assert x % y == 0

    new_shard_sizes = [int(x/y) for x,y in zip(bigm.shard_sizes, breakdowns)]

    X_sharded_new = BigMatrix("reshard({0},{1})".format(bigm.key, breakdowns), bucket=bigm.bucket, shape=bigm.shape, shard_sizes=new_shard_sizes)

    chunked_idxs = []
    chunked_absolute_idxs = []
    for i in range(len(bigm.shape)):
        chunked_idxs.append([tuple(x) for x in matrix_utils.chunk(X_sharded_new._block_idxs(i), breakdowns[i])])
        chunked_absolute_idxs.append([tuple(x) for x in matrix_utils.chunk(X_sharded_new._blocks(i), breakdowns[i])])


    idxs = [bigm._block_idxs(i) for  i in range(len(bigm.shape))]
    all_idxs_new = list(itertools.product(*chunked_idxs))
    all_idxs_old = list(itertools.product(*idxs))
    all_idxs_new_absolute = list(itertools.product(*chunked_absolute_idxs))
    idx_info = list(zip(all_idxs_new, all_idxs_old, all_idxs_new_absolute))


    def reshard_func(bidx_info, bigm, bigm_new):
        idxs_new, idx_old, idx_absolute = bidx_info
        data = bigm.get_block(*idx_old)
        logical = list(itertools.product(*idxs_new))
        absolute = list(itertools.product(*idx_absolute)) 
        offsets = [x[0][0] for x in idx_absolute]
        for lidx, aidx in zip(logical, absolute):
            aidx_offsets = [ slice(x[0] - ox, x[1] - ox)  for x,ox in zip(aidx, offsets)]
            sub_data = data.__getitem__(aidx_offsets)
            print(lidx, aidx, idx_old)
            bigm_new.put_block(sub_data, *lidx)

    if (pwex is None):
        [reshard_func(x, bigm, X_sharded_new) for x in idx_info]
    else:

        futures = pwex.map(lambda x: reshard_func(x, bigm, X_sharded_new), idx_info)
        pywren.wait(futures)
        [f.result() for f in futures]

    return X_sharded_new



