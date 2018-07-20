import boto3
import itertools
import numpy as np
from .matrix import BigMatrix
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
from . import lambdapack as lp
from . import job_runner 


def ind1Dto2D(i, len_A_coded, num_parity_blocks):
    if i < len_A_coded:
        return i//num_parity_blocks, i%num_parity_blocks
    else:
        return i - len_A_coded, num_parity_blocks

def ind2Dto1D(i,j, len_A_coded, num_parity_blocks):
    if j < num_parity_blocks:
        return i*num_parity_blocks + j
    else:
        return i + len_A_coded


def peel_row(y_local, i, bitmask, num_parity_blocks, len_A_coded, shard_size ):
    if bitmask[i, num_parity_blocks] == 0:
        ind = ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks)
        total = y_local[ind*shard_size:(ind + 1)*shard_size]
        for k in range(num_parity_blocks):
            if bitmask[i, k] == 0:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                print("row ind used", ind)
                total = total - y_local[ind*shard_size:(ind + 1)*shard_size]
        a = [ind for (ind, val) in enumerate(bitmask[i]) if val == 1]
        a = a[0]
        print("Filling row singleton", ind2Dto1D(i, a, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(i, a, len_A_coded, num_parity_blocks)
    else:
        total = None
        for k in range(num_parity_blocks):
            if total is None:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                total = y_local[ind * shard_size:(ind + 1) * shard_size]
            else:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                total = total + y_local[ind * shard_size:(ind + 1) * shard_size]
        print("Filling row singleton", ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks)


def peel_col(y_local, j, bitmask, coding_length, len_A_coded, shard_size, num_parity_blocks):
    if bitmask[coding_length, j] == 0:
        ind = ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks)
        total = y_local[ind*shard_size:(ind + 1)*shard_size]
        for k in range(coding_length):
            if bitmask[k, j] == 0:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                print("col ind used", ind)
                total = total - y_local[ind * shard_size:(ind + 1) * shard_size]
        a = [ind for (ind, val) in enumerate(bitmask[:, j]) if val == 1]
        a = a[0]
        print("Filling col singleton", ind2Dto1D(a, j, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(a, j, len_A_coded, num_parity_blocks)
    else:
        total = None
        for k in range(coding_length):
            if total is None:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                total = y_local[ind * shard_size:(ind + 1) * shard_size]
            else:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                total = total + y_local[ind * shard_size:(ind + 1) * shard_size]
        print("Filling col singleton", ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks)


def decode_vector(Y, num_parity_blocks):
    print("DECODING STARTED")
    block_idxs_exist = set([x[0] for x in Y.block_idxs_exist])
    y_local = np.zeros(Y.shape)
    shard_size = Y.shard_sizes[0]
    vector_length_blocks = int(num_parity_blocks*(Y.shape[0]//shard_size - 1 - num_parity_blocks))//(num_parity_blocks+1)
    coding_length = vector_length_blocks // num_parity_blocks
    len_A_coded = vector_length_blocks + num_parity_blocks
    # if Y.shape[0] == coding_length*num_parity_blocks*shard_size:
    #     bitmask = np.ones((coding_length, num_parity_blocks))
    # else:
    bitmask = np.ones((coding_length + 1, num_parity_blocks + 1))
    for r in block_idxs_exist:
        y_local[r*shard_size:(r + 1)*shard_size] = Y.get_block(r, 0)
        i, j = ind1Dto2D(r, len_A_coded, num_parity_blocks)
        bitmask[i, j] = 0
    print("BITmask\n", bitmask)
    # print("y_local before decoding\n", np.isclose(y_local, bk_local))

    while (bitmask.sum() > 0):
        row_sum = bitmask.sum(axis=1)
        r = [ind for (ind, val) in enumerate(row_sum) if val == 1]
        print("row singletons", r)
        for rr in r:
            y_local_block, ind = peel_row(y_local, rr, bitmask, num_parity_blocks, len_A_coded, shard_size)
            y_local[ind * shard_size:(ind + 1) * shard_size] = y_local_block
        bitmask[r] = 0

        col_sum = bitmask.sum(axis=0)
        c = [ind for (ind, val) in enumerate(col_sum) if val == 1]
        print("col singletons", c)
        for cc in c:
            y_local_block,ind = peel_col(y_local, cc, bitmask, coding_length, len_A_coded, shard_size, num_parity_blocks)
            y_local[ind * shard_size:(ind + 1) * shard_size] = y_local_block
        bitmask[:, c] = 0
    # print ("y_decoded\n", np.isclose(y_local, bk_local))
    y_local = y_local[0:vector_length_blocks*shard_size]
    return y_local

def _gemm_remote_3(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    print(reduce_idxs)
    print(block_pairs)
    t1 = time.time()
    assert(len(Y.shape) == 1 or Y.shape[1] == 1)
    num_parity_blocks = kwargs['num_parity_blocks']
    breakdown = kwargs['breakdown']
    print("1", time.time() - t1)
    if num_parity_blocks==0:
        shard_size = Y.shard_sizes[0]
        y_local = np.zeros(Y.shape)
        # if Y.block_idxs_not_exist!=[]:
        #     print("ERROR, some blocks in bk missing")
        #     return
        # block_idxs_exist = set([x[0] for x in Y.block_idxs_exist])
        # for r in block_idxs_exist:
        print("2", time.time() - t1)
        for r in Y._block_idxs(0):
            y_local[r*shard_size:(r + 1)*shard_size] = Y.get_block(r, 0)
            if r%50==0:
                print("r, time", r, time.time() - t1)
    else:
        y_local = decode_vector(Y, num_parity_blocks)
    # print("y_local after decoding", y_local)
    
    t2 = time.time()
    print ("phase 1 time", t2-t1)
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XY_block = None
        X.dtype = dtype
        for r in reduce_idxs:
            print ("At reduce id:", r)
            block1 = X.get_block(bidx_0, r)
            sidx,eidx = Y.blocks[r]
            sidx, eidx = sidx
            sidx = int(sidx*breakdown)
            eidx = int(eidx*breakdown)
            y_block = y_local[sidx:eidx]
            # print ("r, y_block\n", r, y_block)
            if (XY_block is None):
                XY_block = block1.dot(y_block)
            else:
                XY_block = XY_block + block1.dot(y_block)
        XY.put_block(XY_block, bidx_0, bidx_1)
    print ("phase 2 time", time.time()-t2)
    return 0


def _gemm_remote_0(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    print(reduce_idxs)
    print(block_pairs)
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XY_block = None
        X.dtype = dtype
        Y.dtype = dtype
        for r in reduce_idxs:
            print ("At reduce id:", r)
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(r, bidx_1)
            if (XY_block is None):
                XY_block = block1.dot(block2)
            else:
                XY_block += block1.dot(block2)
        # print("block1-block2 shape", block1.shape, block2.shape)
        # print("output block shape", XY_block.shape)
        XY.put_block(XY_block, bidx_0, bidx_1)

def _gemm_remote_1(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    os.system("sudo mount -o remount,size=50g /dev/shm")
    X.dtype = dtype
    Y.dtype = dtype
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        block0 = matrix_utils.get_row(X, bidx_0, mmap_loc="/dev/shm/block_0")
        block1 = matrix_utils.get_col(Y, bidx_1, mmap_loc="/dev/shm/block_1")
        XY_block = block0.dot(block1)
        XY.put_block(XY_block, bidx_0, bidx_1)

def _gemm_remote_2(block_pairs, XY, X, Y, reduce_idxs=[0], dtype=np.float64, **kwargs):
    os.system("sudo mount -o remount,size=50g /dev/shm")
    X.dtype = dtype
    X.dtype = dtype
    Y.dtype = dtype
    block_chunk_size = kwargs.get("block_chunk_size")
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        result = gemm_with_prefetch(X, Y, bidx_0, bidx_1, block_chunk_size=block_chunk_size)
        XY.put_block(result, bidx_0, bidx_1)

_gemms = [_gemm_remote_0, _gemm_remote_1, _gemm_remote_2, _gemm_remote_3]


def gemm_with_prefetch(X, Y, bidx0, bidx1, block_chunk_size=16):
    # prefetch first 16 columns 
    parity = 0
    executor = fs.ProcessPoolExecutor(32)
    futures0 = matrix_utils.get_matrix_blocks_full_async(X, "/dev/shm/block0_{0}".format(parity), [bidx0], list(range(block_chunk_size)), big_axis=1, executor=executor)
    futures1 = matrix_utils.get_matrix_blocks_full_async(Y, "/dev/shm/block1_{0}".format(parity), list(range(block_chunk_size)), [bidx1], big_axis=0, executor=executor)
    assert X._block_idxs(1) == Y._block_idxs(0)
    chunked_blocks = list(matrix_utils.chunk(X._block_idxs(1), block_chunk_size))
    assert(chunked_blocks[0] == list(range(block_chunk_size)))
    chunked_blocks = chunked_blocks[1:]
    start_x, end_x = X._blocks(0)[bidx0]
    start_y, end_y = Y._blocks(1)[bidx1]
    result = np.zeros((end_x - start_x, end_y - start_y), dtype=X.dtype)
    for blocks in chunked_blocks:
        t = time.time()
        fs.wait(futures0)
        fs.wait(futures1)
        e = time.time()
        print("Block Download took effectively {0}".format(e - t))
        results = [f.result() for f in futures0]
        b1 = matrix_utils.load_mmap(*results[0])
        results = [f.result() for f in futures1]
        b2 = matrix_utils.load_mmap(*results[0])
        parity = (parity + 1) % 2
        futures0 = matrix_utils.get_matrix_blocks_full_async(X, "/dev/shm/block0_{0}".format(parity), [bidx0], blocks, big_axis=1, executor=executor)
        futures1 = matrix_utils.get_matrix_blocks_full_async(Y, "/dev/shm/block1_{0}".format(parity), blocks, [bidx1], big_axis=0, executor=executor)
        t = time.time()
        result += b1.dot(b2)
        e = time.time()
        print("Block Matmul took effectively {0}".format(e  - t))
    t = time.time()
    fs.wait(futures0)
    fs.wait(futures1)
    e = time.time()
    print("Block Download took effectively {0}".format(e - t))
    results = [f.result() for f in futures0]
    b1 = matrix_utils.load_mmap(*results[0])
    results = [f.result() for f in futures1]
    b2 = matrix_utils.load_mmap(*results[0])
    t = time.time()
    result += b1.dot(b2)
    e = time.time()
    print("Block Matmul took effectively {0}".format(e  - t))
    return result


def gemm(pwex, X, Y, out_bucket=None, tasks_per_job=1, local=False, dtype=np.float64, overwrite=True, gemm_impl=0, gemm_chunk_size=16, straggler_thresh=1.0, parent_fn=None, **kwargs):

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

    reduce_idxs = X._block_idxs(axis=1)
    if (out_bucket == None):
        out_bucket = X.bucket
    root_key = generate_key_name_binop(X, Y, "gemm")
    if len(root_key)>200:
        root_key = matrix_utils.hash_string(root_key)
    print("Output key:", root_key)
    if (Y.shard_sizes[0] !=  X.shard_sizes[1] and gemm_impl!=3):
        raise Exception("X dim 1 shard size must match Y dim 0 shard size")
    XY = BigMatrix(root_key, shape=(X.shape[0], Y.shape[1]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], Y.shard_sizes[1]], dtype=dtype, parent_fn=parent_fn, autosqueeze=False, write_header=True)


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
    chunked_blocks = chunk(block_idxs_to_map, tasks_per_job)
    if (not isinstance(pwex.invoker, pywren.queues.SQSInvoker) and gemm_impl > 0 and gemm_impl < 3):
            raise Exception("GEMM IMPL > 0 only supported for standalone mode pywren")

    print(_gemms[gemm_impl])
    def pywren_run(x):
        return _gemms[gemm_impl](x, XY, X, Y, reduce_idxs=reduce_idxs, dtype=dtype, block_chunk_size=gemm_chunk_size, **kwargs)

    if (local):
        list(map(pywren_run, chunked_blocks))
    else:
        s = time.time()
        futures = pwex.map(pywren_run, chunked_blocks)
        e = time.time()
        print("Pwex Map Time {0}".format(e - s))
        print("Number of futures:", len(futures))
    if (local):
        return XY
    while (True):
        fs_dones, fs_notdones = pywren.wait(futures, 3)
        result_count = len(fs_dones)
        print("workers done, time passed since mapping", result_count, time.time() - e)
        if (result_count >= straggler_thresh*len(futures)):
            # [f.result() for f in fs_dones]
            for f in fs_dones:
                try:
                    f.result()
                except Exception as e:
                    print(e)
                    pass
            break
        time.sleep(2)
        # print("Time passed since mapping", time.time() - e)
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

def trisolve(pwex, A, B, out_bucket=None, tasks_per_job=1, lower=False):
    if (out_bucket == None):
        out_bucket = A.bucket

    root_key = generate_key_name_binop(A, B, "trisolve")
    instructions, X, scratch = lp._trisolve(A, B, out_bucket=out_bucket, lower=lower)
    config = pwex.config
    # if (isinstance(pwex.invoker, pywren.queues.SQSInvoker)):
    #     executor = pywren.standalone_executor
    # else:
    executor = pywren.lambda_executor
    program = lp.LambdaPackProgram(instructions, executor=executor, pywren_config=config)
    print(program)
    #assert False
    program.start()
    job_runner.lambdapack_run(program)
    program.wait()
    if program.program_status() != lp.PS.SUCCESS:
        program.unwind()
        raise Exception("Lambdapack Exception : {0}".format(program.program_status()))
    program.free()

    # delete all intermediate information
    [M.free() for M in scratch] 
    return X 

