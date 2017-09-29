import concurrent.futures as fs
import io
import itertools
import os
import time

import boto3
import cloudpickle
import numpy as np
import hashlib

def hash_string(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if n == 0: return []
    for i in range(0, len(l), n):
        yield l[i:i + n]

def generate_key_name(X, Y, op):
    assert op == "cxyt"

    if (X.key == Y.key):
        key = "XXT({0})".format(X.key)
    else:
        key = "XYT({0}, {1})".format(X.key, Y.key)
    return key


def list_all_keys(bucket, prefix):
    client = boto3.client('s3')
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    keys = list(map(lambda x: x['Key'], objects['Contents']))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))

def block_key_to_block(key):
    try:
        block_key = key.split("/")[-1]
        blocks_split = block_key.split("_")
        b0_start = int(blocks_split[0])
        b0_end = int(blocks_split[1])
        b1_start = int(blocks_split[3])
        b1_end = int(blocks_split[4])
        return ((b0_start, b0_end), (b1_start, b1_end))
    except Exception as e:
        return None

def get_local_matrix(X_sharded, dtype="float64", workers=22, mmap_loc=None):
    hash_key = hash_string(X_sharded.key)
    if (mmap_loc == None):
        mmap_loc = "/tmp/{0}".format(hash_key)
    return fast_kernel_column_blocks_get(X_sharded, \
                                  col_blocks=X_sharded._block_idxs(1), \
                                  row_blocks=X_sharded._block_idxs(0), \
                                  mmap_loc=mmap_loc, \
                                  workers=workers, \
                                  dtype=dtype)

def fast_kernel_column_blocks_get(K, col_blocks, mmap_loc, workers=21, dtype="float64", row_blocks=None):
    futures = fast_kernel_column_block_async(K, col_blocks, mmap_loc=mmap_loc, workers=workers, dtype=dtype, row_blocks=row_blocks)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

def fast_kernel_column_block_async(K, col_blocks, executor=None, workers=23, mmap_loc="/dev/shm/block0", wait=False, dtype="float64", row_blocks=None):

    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)


    if (row_blocks == None):
        row_blocks = K._block_idxs(0)

    total_block_width = 0
    total_block_height = 0
    for row_block in row_blocks:
        total_block_height += min(K.shard_sizes[0], K.shape[0] - row_block*K.shard_sizes[0])

    for col_block in col_blocks:
        total_block_width += min(K.shard_sizes[1], K.shape[1] - col_block*K.shard_sizes[1])

    mmap_shape = (total_block_height,  total_block_width)
    s = time.time()
    np.memmap(mmap_loc, dtype=dtype, mode='w+', shape=mmap_shape)
    e = time.time()
    futures = []
    chunk_size = int(np.ceil(len(row_blocks)/workers))
    chunks = chunk(row_blocks, chunk_size)
    row_offset = 0
    for c in chunks:
        futures.append(executor.submit(K.get_blocks_mmap, c, col_blocks, mmap_loc, mmap_shape, dtype=dtype, row_offset=row_offset, col_offset=0))
        row_offset += len(c)
    return futures

def fast_kernel_row_block_async(K, col_blocks, executor=None, workers=23, mmap_loc="/dev/shm/block0", wait=False, dtype="float64", row_blocks=None):

    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)


    if (row_blocks == None):
        row_blocks = K._block_idxs(0)

    total_block_width = 0
    total_block_height = 0
    for row_block in row_blocks:
        total_block_height += min(K.shard_sizes[0], K.shape[0] - row_block*K.shard_sizes[0])

    print(col_blocks)
    print(max(col_blocks))
    print("ARGMAX", np.argmax(col_blocks))
    for col_block in col_blocks:
        total_block_width += min(K.shard_sizes[1], K.shape[1] - col_block*K.shard_sizes[1])

    mmap_shape = (total_block_height,  total_block_width)
    print("MMAP SHAPE IS ", mmap_shape)
    s = time.time()
    X = np.memmap(mmap_loc, dtype=dtype, mode='w+', shape=mmap_shape)
    e = time.time()
    futures = []
    chunk_size = int(np.ceil(len(col_blocks)/workers))
    chunks = misc.chunk(col_blocks, chunk_size)
    col_offset = 0
    for c in chunks:
        futures.append(executor.submit(K.get_blocks_mmap, row_blocks, c, mmap_loc, mmap_shape, dtype=dtype, col_offset=col_offset, row_offset=0))
        col_offset += len(c)
    return futures

def fast_kernel_column_blocks_get(K, col_blocks, mmap_loc, workers=21, dtype="float64", row_blocks=None):
    futures = fast_kernel_column_block_async(K, col_blocks, mmap_loc=mmap_loc, workers=workers, dtype=dtype, row_blocks=row_blocks)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

def fast_kernel_row_blocks_get(K, col_blocks, mmap_loc, workers=21, dtype="float64", row_blocks=None):
    futures = fast_kernel_row_block_async(K, col_blocks, mmap_loc=mmap_loc, workers=workers, dtype=dtype, row_blocks=row_blocks)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())


def load_mmap(mmap_loc, mmap_shape, mmap_dtype):
    return np.memmap(mmap_loc, dtype=mmap_dtype, mode='r+', shape=mmap_shape)

