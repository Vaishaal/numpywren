from .matrix import BigMatrix, BigSymmetricMatrix
import numpy as np
import concurrent.futures as fs
import pywren
from .matrix_utils import generate_key_name_uop, constant_zeros
from .matrix_init import shard_matrix, local_numpy_init
import pytest
from sys import getsizeof
import time


class RemoteLoad(object):
    def __init__(self, i_id, matrix, *bidxs):
        self.id = i_id
        self.matrix = matrix
        self.bidxs = bidxs
        self.result = None

    def __call__(self):
        if (self.result == None):
            self.result = self.matrix.get_block(*self.bidxs)
        return self.result

    def __str__(self):
        return "{0} = S3_LOAD {1} {2}".format(self.id, self.matrix, self.bidxs)

class RemoteWrite(object):
    def __init__(self, i_id, matrix, data_instr, *bidxs):
        self.id = i_id
        self.matrix = matrix
        self.bidxs = bidxs
        self.data_instr = data_instr
        self.result = None

    def __call__(self):
        if (self.result == None):
            self.result = self.matrix.put_block(data_instr.result, *self.bidxs)
        return self.result

    def __str__(self):
        return "{0} = S3_WRITE {1} {2} {3}".format(self.id, self.matrix, self.bidxs, self.data_instr.id)

class RemoteSYRK(object):
    def __init__(self, i_id, argv_instr):
        self.id = i_id
        assert len(argv_instr) == 3
        self.argv_instr = argv_instr
        self.result = None
    def __call__(self):
        if (self.result == None):
            old_block = argv_instr[0].result
            block_2 = argv_instr[1].result
            block_1 = argv_instr[2].result
            self.result = old_block - block_2.dot(block_1.T)
        return self.result

    def __str__(self):
        return "{0} = SYRK {1} {2} {3}".format(self.id, self.argv[0].id,  self.argv[1].id,  self.argv[2].id)

class RemoteTRSM(object):
    def __init__(self, i_id, argv_instr):
        self.id = i_id
        assert len(argv_instr) == 2
        self.argv_instr = argv_instr
        self.result = None
    def __call__(self):
        if (self.result == None):
            L_bb_inv = argv_instr[0].result
            col_block = argv_instr[1].result
            self.result = col_block.T.dot(L_bb_inv)
        return self.result

    def __str__(self):
        return "{0} = TRSM {1} {2}".format(self.id, self.argv[0].id,  self.argv[1].id)

class RemoteCholesky(object):
    def __init__(self, i_id, argv_instr):
        self.id = i_id
        assert len(argv_instr) == 1
        self.argv_instr = argv_instr
        self.result = None
    def __call__(self):
        if (self.result == None):
            L_bb = argv_instr[0].result
            self.result = np.linalg.cholesky(L_bb)
        return self.result

    def __str__(self):
        return "{0} = CHOL {1}".format(self.id, self.argv[0].id)

class RemoteInverse(object):
    def __init__(self, i_id, argv_instr):
        self.id = i_id
        assert len(argv_instr) == 1
        self.argv_instr = argv_instr
        self.result = None
    def __call__(self):
        if (self.result == None):
            L_bb = argv_instr[0].result
            self.result = np.linalg.cholesky(L_bb)
        return self.result

    def __str__(self):
        return "{0} = INVRS {1}".format(self.id, self.argv[0].id)


def make_block_matmul_update(pc, L_out, L_in, L_bb_inv, b0, b1):
    L_load = RemoteLoad(pc, L_in, b0, b1)
    pc += 1
    L_bb_inv_load = RemoteLoad(pc, L_bb_inv, 0, 0)
    pc += 1
    trsm = RemoteTRSM(pc, [L_load, L_bb_inv_load])
    pc += 1
    write = RemoteWrite(pc, L_out, trsm, b0, b1)
    return [L_load, L_bb_inv_load, trsm, write]

def make_low_rank_update(pc, L_out, L_in, b0, b1, b2):
    old_block_load = RemoteLoad(pc, L_in, b1, b2)
    pc += 1
    block_1_load = RemoteLoad(pc, L_in, b0, b1)
    pc += 1
    block_2_load = RemoteLoad(pc, L_in, b0, b1)
    pc += 1
    syrk = RemoteSYRK(pc, [old_block_load, block_1_load, block_2_load])
    pc += 1
    write = RemoteWrite(pc, L_out, syrk, b1, b2)
    return [old_block_load, block_1_load, block_2_load, syrk, write]

def make_local_cholesky_and_inverse(pc, L_out, L_bb_inv_out, L_in, b0):
    block_load = RemoteLoad(pc, L_in, b0, b0)
    cholesky = RemoteCholesky(pc, [block_load])
    inverse = RemoteInverse(pc, [cholesky])
    write_diag = RemoteWrite(pc, L_out, cholesky, b0, b0)
    write_inverse = RemoteWrite(pc, L_bb_inv_out, inverse, 0, 0)
    return [block_load, cholesky, inverse, write_diag, write_inverse]


def block_matmul_update(L, X, L_bb_inv, block_0_idx, block_1_idx):
    read_start = time.time()
    L_bb_inv = L_bb_inv.get_block(0,0)
    X_block = X.get_block(block_1_idx, block_0_idx).T
    read_time = time.time() - read_start
    read_size = getsizeof(L_bb_inv) + getsizeof(X_block)
    compute_start = time.time()
    L_block = X_block.T.dot(L_bb_inv)
    compute_time = time.time() - compute_start
    flops = X.shape[0]*X.shape[1]*L_bb_inv.shape[1]*2
    write_start = time.time()
    L.put_block(L_block, block_1_idx, block_0_idx)
    write_time = time.time() - write_start
    write_size = getsizeof(L_block)
    return  read_time, compute_time, write_time, flops, read_size, write_size




def syrk_update(L, X, block_0_idx, block_1_idx, block_2_idx):
    read_start = time.time()
    block_1 = L.get_block(block_1_idx, block_0_idx)
    block_2 = L.get_block(block_2_idx, block_0_idx)
    old_block = X.get_block(block_2_idx, block_1_idx)
    X_block = X.get_block(block_1_idx, block_2_idx).T
    read_time = time.time() - read_start
    read_size = getsizeof(block_1) + getsizeof(block_2) + getsizeof(old_block) + getsizeof(X_block)

    compute_start = time.time()
    update = old_block - block_2.dot(block_1.T)
    compute_time = time.time() - compute_start

    write_start = time.time()
    L.put_block(update, block_2_idx, block_1_idx)
    write_time =  time.time() - write_start
    flops = block_2.shape[0] * block_2.shape[1] * block_1.shape[1] * 2 + old_block.shape[0]*old_block.shape[1]
    write_size = getsizeof(update)
    return read_time, compute_time, write_time, flops, read_size, write_size




def chol(pwex, X, out_bucket=None, tasks_per_job=1):
    if (out_bucket == None):
        out_bucket = X.bucket
    out_key = generate_key_name_uop(X, "chol")
    # generate output matrix
    L = BigMatrix(out_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]], parent_fn=constant_zeros)
    # generate intermediate matrices
    trailing_matrices = [X]
    cholesky_diag_inverses = []
    all_blocks = list(L.block_idxs)
    block_idxs = X._block_idxs(0)

    for i,j0 in enumerate(X._block_idxs(0)):
        L_trailing = BigMatrix(out_key + "_{0}_trailing".format(i),
                       shape=(X.shape[0], X.shape[0]),
                       bucket=out_bucket,
                       shard_sizes=[X.shard_sizes[0], X.shard_sizes[0]],
                       parent_fn=constant_zeros)
        if (j0 == X._block_idxs(0)[-1]):
            block_size =  X.shard_sizes[0]
        else:
            block_size = X.shape[0] - X.shard_sizes[0]*j0
        L_bb_inv = BigMatrix(out_key + "_{(0},{0})_inv".format(i),
                             shape=(block_size, block_size),
                             bucket=out_bucket,
                             shard_sizes=[block_size, block_size],
                             parent_fn=constant_zeros)

        trailing_matrices.append(L_trailing)
        cholesky_diag_inverses.append(L_bb_inv)
    trailing_matrices.append(L)
    all_instructions = []

    pc = 0
    for i in block_idxs:
        L_inv = cholesky_diag_inverses[i]
        instructions, count = make_local_cholesky_and_inverse(pc, trailing[-1], trailing[i], L_inv, i)
        all_instructions += instructions
        for j in block_idxs:
            if (j > i): continue
            isntructions, count = make_column_update(pc, trailing[-1], trailing[i], L_inv, i, j)
            all_instructions += instructions
        for j in block_idxs:
            if (j > i): continue
            for k in block_idxs:
                if (k > j): continue
                instructions, count = make_low_rank_update(pc, trailing[i+1], trailing[i], i, j, k)
                all_instructions += instructions
    for inst in all_instructions:
        print(inst)



