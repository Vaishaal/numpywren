import concurrent.futures as fs
import io
import itertools
import os
import time

import boto3
import botocore
import cloudpickle
import numpy as np
import hashlib
import pickle
import pywren.serialize as serialize
import inspect
from . import matrix
import multiprocessing

cpu_count = multiprocessing.cpu_count()

class MmapArray():
    def __init__(self, mmaped, mode=None,idxs=None):
        self.loc = mmaped.filename
        self.dtype = mmaped.dtype
        self.shape = mmaped.shape
        self.mode = mmaped.mode
        self.idxs = idxs
        if (mode != None):
            self.mode = mode

    def load(self):
        X = np.memmap(self.loc, dtype=self.dtype, mode=self.mode, shape=self.shape)
        if self.idxs != None:
            return X[self.idxs[0]:self.idxs[1]]
        else:
            return X

def hash_string(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def hash_array(s):
    s = np.ascontiguousarray(s)
    byte_view = s.view(np.uint8)
    return hashlib.sha1(byte_view).hexdigest()

def hash_function(f):
    src_code = inspect.getsource(f)
    return hashlib.sha1(src_code.encode()).hexdigest()

def hash_bytes(byte_string):
    return hashlib.sha1(byte_string.encode('utf-8')).hexdigest()

def hash_args(args):
    arg_bytes = pickle.dumps(args)
    return hashlib.sha1(arg_bytes).hexdigest()

def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if n == 0: return []
    for i in range(0, len(l), n):
        yield l[i:i + n]

def generate_key_name_binop(X, Y, op):
    assert op == "gemm"
    if (op == "gemm"):
        key = "gemm({0}, {1})".format(str(X), str(Y))
    return key

def generate_key_name_uop(X, op):
    assert op == "chol"
    if (op == "chol"):
        key = "chol({0})".format(str(X))
        return key

def generate_key_name_local_matrix(X_local):
    return hash_array(X_local)

def load_mmap(mmap_loc, mmap_shape, mmap_dtype):
    return np.memmap(mmap_loc, dtype=mmap_dtype, mode='r+', shape=mmap_shape)

def list_all_keys(bucket, prefix):
    client = boto3.client('s3')
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    if (objects.get('Contents') == None):
        return []
    keys = list(map(lambda x: x['Key'], objects.get('Contents', [] )))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))

def key_exists(bucket, key):
    '''Return true if a key exists in s3 bucket'''
    client = boto3.client('s3')
    try:
        obj = client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise
        return False

def block_key_to_block(key):
    try:
        block_key = key.strip().split("/")[-1]
        if (block_key) == "header": return None
        blocks_split = block_key.strip('_').split("_")

        assert(len(blocks_split)%3 == 0)
        block = []
        for i in range(0,len(blocks_split),3):
            start = int(blocks_split[i])
            end = int(blocks_split[i+1])
            block.append((start,end))

        return tuple(block)
    except Exception as e:
        raise
        return None

def get_blocks_mmap(bigm, block_idxs, local_idxs, mmap_loc, mmap_shape):
    '''Map block_idxs to local_idxs in np.memmamp object found at mmap_loc'''
    X_full = np.memmap(mmap_loc, dtype=bigm.dtype, mode='r+', shape=mmap_shape)
    for block_idx, local_idx  in zip(block_idxs, local_idxs):
        local_idx_slices = [slice(s,e) for s,e in local_idx]
        block_data = bigm.get_block(*block_idx)
        X_full[local_idx_slices] = block_data
    X_full.flush()
    return (mmap_loc, mmap_shape, bigm.dtype)


def get_local_matrix(bigm, workers=cpu_count, mmap_loc=None, big_axis=0):
    hash_key = hash_string(bigm.key)
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)

    #if (os.path.isfile(mmap_loc)):
    #    return np.memmap(mmap_loc, dtype=bigm.dtype, mode="r+", shape=tuple(bigm.shape))

    executor = fs.ProcessPoolExecutor(max_workers=workers)
    blocks_to_get = [bigm._block_idxs(i) for i in range(len(bigm.shape))]
    big_axis = np.argmax([len(bigm._block_idxs(i)) for i in range(len(bigm.shape))])
    print("big axis", big_axis)
    futures = get_matrix_blocks_full_async(bigm, mmap_loc, *blocks_to_get, big_axis=big_axis)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

## TODO: generalize for arbitrary MD arrays 
def get_col(bigm, col, workers=cpu_count, mmap_loc=None):
    assert len(bigm.shape) == 2
    hash_key = hash_string(bigm.key)
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    executor = fs.ProcessPoolExecutor(max_workers=workers)
    print(bigm.block_idxs)
    blocks_to_get = [bigm._block_idxs(0), [col]]
    print(blocks_to_get)
    futures = get_matrix_blocks_full_async(bigm, mmap_loc, *blocks_to_get, executor=executor, big_axis=0)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())


def put_col_async(bigm, mmap_loc, shape, block, bidx):
    X_memmap = np.memmap(mmap_loc, dtype=bigm.dtype, mode='r+', shape=shape)
    block_data = X_memmap[block[0][0]:block[0][1], :]
    bigm.put_block(block_data, *bidx)
    return 0

def put_col(bigm, col, workers=cpu_count, mmap_loc=None, big_axis=0):
    assert len(bigm.shape) == 2
    hash_key = hash_string(bigm.key + str(col))
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    executor = fs.ProcessPoolExecutor(max_workers=workers)
    block_idx_blocks = list(zip(bigm.block_idxs, bigm.blocks))
    blocks_to_put = [x for x in block_idx_blocks if x[0][1] == col]

    X_memmap = np.memmap(mmap_loc, dtype=bigm.dtype, mode='w+', shape=col.shape)
    futures = []
    for bidx, block  in blocks_to_put:
        futures.append(executor.submit(put_col_async, bigm, mmap_loc, col.shape, block, bidx))
    fs.wait(futures)
    [f.result() for f in futures]
    return



## TODO: generalize for arbitrary MD arrays 
def get_row(bigm, row, workers=cpu_count, mmap_loc=None):
    assert len(bigm.shape) == 2
    hash_key = hash_string(bigm.key)
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    executor = fs.ProcessPoolExecutor(max_workers=workers)
    blocks_to_get = [[row], bigm._block_idxs(1)]
    futures = get_matrix_blocks_full_async(bigm, mmap_loc, *blocks_to_get, executor=executor, big_axis=1)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

def put_row_async(bigm, mmap_loc, shape, block, bidx):
    X_memmap = np.memmap(mmap_loc, dtype=bigm.dtype, mode='r+', shape=shape)
    block_data = X_memmap[:, block[1][0]:block[1][1]]
    bigm.put_block(block_data, *bidx)
    return 0

def put_row(bigm, data, row, workers=cpu_count, mmap_loc=None, big_axis=0):
    assert len(bigm.shape) == 2
    hash_key = hash_string(bigm.key + str(row))
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    executor = fs.ProcessPoolExecutor(max_workers=workers)
    block_idx_blocks = list(zip(bigm.block_idxs, bigm.blocks))
    blocks_to_put = [x for x in block_idx_blocks if x[0][0] == row]

    X_memmap = np.memmap(mmap_loc, dtype=bigm.dtype, mode='w+', shape=data.shape)
    np.copyto(X_memmap, data)
    futures = []
    for bidx, block in blocks_to_put:
        futures.append(executor.submit(put_row_async, bigm, mmap_loc, data.shape, block, bidx))
    fs.wait(futures)
    [f.result() for f in futures]
    return

def get_matrix_blocks_full_async(bigm, mmap_loc, *blocks_to_get, big_axis=0, executor=None, workers=cpu_count):
    '''
        Download blocks from bigm using multiprocess and memmap to maximize S3 bandwidth
        * blocks_to_get is a list equal in length to the number of dimensions of bigm
        * each element of that list is a block to get from that axis
    '''
    mmap_shape = []
    local_idxs = []
    matrix_locations = [{} for _ in range(len(bigm.shape))]
    matrix_maxes = [0 for _ in range(len(bigm.shape))]
    current_local_idx = np.zeros(len(bigm.shape), np.int)
    # statically assign parts of our mmap matrix to parts of our sharded matrix
    for axis, axis_blocks in enumerate(blocks_to_get):
        axis_size = 0
        for block in axis_blocks:
            size = int(min(bigm.shard_sizes[axis], bigm.shape[axis] - block*bigm.shard_sizes[axis]))
            axis_size += size
            start = bigm.shard_sizes[axis]*block
            end = start + size
            if (matrix_locations[axis].get((start,end)) == None):
                matrix_locations[axis][(start,end)] = (matrix_maxes[axis], matrix_maxes[axis]+size)
                matrix_maxes[axis] += size
        mmap_shape.append(axis_size)

    mmap_shape = tuple(mmap_shape)
    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)
    np.memmap(mmap_loc, dtype=bigm.dtype, mode='w+', shape=mmap_shape)
    futures = []

    # chunk across whatever we decided is our "big axis"
    chunk_size = int(np.ceil(len(blocks_to_get[big_axis])/workers))
    chunks = list(chunk(blocks_to_get[big_axis], chunk_size))
    blocks_to_get = list(blocks_to_get)
    for c in chunks:
        c = sorted(c)
        small_axis_blocks = blocks_to_get.copy()
        del small_axis_blocks[big_axis]
        small_axis_blocks.insert(big_axis, c)
        block_idxs = list(itertools.product(*small_axis_blocks))
        local_idxs = []
        for block_idx in block_idxs:
            real_idx = bigm.__block_idx_to_real_idx__(block_idx)
            local_idx = tuple((matrix_locations[i][(s,e)] for i,(s,e) in enumerate(real_idx)))
            local_idxs.append(local_idx)
        futures.append(executor.submit(get_blocks_mmap, bigm, block_idxs, local_idxs, mmap_loc, mmap_shape))
    return futures

def make_constant_parent(cnst):
    def constant_parent(bigm, *block_idx):
        real_idxs = bigm.__block_idx_to_real_idx__(block_idx)
        current_shape = tuple([e - s for s,e in real_idxs])
        return np.full(current_shape, cnst)
    return constant_parent


def constant_zeros(bigm, *block_idx):
    real_idxs = bigm.__block_idx_to_real_idx__(block_idx)
    current_shape = tuple([e - s for s,e in real_idxs])
    return np.zeros(current_shape)

















