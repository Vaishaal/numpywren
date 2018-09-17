import numpy as np
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren import lambdapack as lp
from numpywren import job_runner, frontend
from numpywren.compiler import lpcompile_for_execution
from numpywren.algs import CHOLESKY, TSQR, GEMM, QR, BDFAC
from numpywren.matrix_utils import constant_zeros, constant_zeros_ext
from numpywren.matrix_init import shard_matrix
import dill
import numpywren as npw
import time



def cholesky(X, truncate=0):
    S = BigMatrix("Cholesky.Intermediate({0})".format(X.key), shape=(X.num_blocks(1)+1, X.shape[0], X.shape[0]), shard_sizes=(1, X.shard_sizes[0], X.shard_sizes[0]), bucket=X.bucket, write_header=True, parent_fn=constant_zeros)
    #S.free()
    O = BigMatrix("Cholesky({0})".format(X.key), shape=(X.shape[0], X.shape[0]), shard_sizes=(X.shard_sizes[0], X.shard_sizes[0]), write_header=True, parent_fn=constant_zeros)
    t = time.time()
    p0= lpcompile_for_execution(CHOLESKY, inputs=["I"], outputs=["O"])
    p1 = p0(O,X,S,int(np.ceil(X.shape[0]/X.shard_sizes[0])), truncate)
    e = time.time()
    c_time = e - t
    config = npw.config.default()
    program = lp.LambdaPackProgram(p1, config=config)
    return program, {"outputs":[O], "intermediates": [S], "compile_time": c_time}


def tsqr(X, truncate=0):
    b_fac = 2
    assert(X.shard_sizes[1] == X.shape[1])
    shard_size = X.shard_sizes[0]
    shard_sizes = X.shard_sizes
    num_tree_levels = max(int(np.ceil(np.log2(X.num_blocks(0))/np.log2(b_fac))), 1)
    R_sharded= BigMatrix("tsqr_R({0})".format(X.key), shape=(num_tree_levels*shard_size, X.shape[0]), shard_sizes=shard_sizes, write_header=True, safe=False)
    T_sharded= BigMatrix("tsqr_T({0})".format(X.key), shape=(num_tree_levels*shard_size*b_fac, X.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)
    V_sharded= BigMatrix("tsqr_V({0})".format(X.key), shape=(num_tree_levels*shard_size*b_fac, X.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=True, safe=False)
    t = time.time()
    p0 = lpcompile_for_execution(TSQR, inputs=["A"], outputs=["Rs"])
    config = npw.config.default()
    N_blocks = X.num_blocks(0)
    p1 = p0(X, V_sharded, T_sharded, R_sharded, N_blocks)
    e = time.time()
    c_time = e - t
    program = lp.LambdaPackProgram(p1, config=config)
    return program, {"outputs":[R_sharded, V_sharded, T_sharded], "intermediates": [], "compile_time": c_time}

def gemm(A, B):
    b_fac = 4
    assert(A.shape[1] == B.shape[0])
    assert(A.shard_sizes[1] == B.shard_sizes[0])
    shard_sizes = (A.shard_sizes[0], B.shard_sizes[1])
    num_tree_levels = max(int(np.ceil(np.log2(A.num_blocks(1))/np.log2(b_fac))), 1)
    Temp = BigMatrix(f"matmul_test_Temp({A.key},{B.key})", shape=(A.shape[0], B.shape[1], B.shape[0], num_tree_levels), shard_sizes=[A.shard_sizes[0], B.shard_sizes[1], 1, 1], write_header=True, safe=False, parent_fn=constant_zeros)
    C_sharded= BigMatrix("matmul_test_C", shape=(A.shape[0], B.shape[1]), shard_sizes=shard_sizes, write_header=True)
    config = npw.config.default()
    t = time.time()
    p0 = lpcompile_for_execution(GEMM, inputs=["A", "B"], outputs=["Out"])
    print("tree depth", np.ceil(np.log(B.num_blocks(1))/np.log(4)))
    p1 = p0(A, B, A.num_blocks(0), A.num_blocks(1), B.num_blocks(1), Temp, C_sharded)
    e = time.time()
    c_time = e - t
    program = lp.LambdaPackProgram(p1, config=config)
    return program, {"outputs":[C_sharded], "intermediates":[Temp], "compile_time": c_time}

def qr(A):
    b_fac = 2
    N = A.shape[0]
    N_blocks = A.num_blocks(0)
    b_fac = 2
    shard_size = A.shard_sizes[0]
    num_tree_levels = max(int(np.ceil(np.log2(A.num_blocks(0))/np.log2(b_fac))), 1) + 1
    Vs = BigMatrix("Vs", shape=(2*N, 2*N, num_tree_levels), shard_sizes=(shard_size, shard_size, 1), write_header=True, parent_fn=constant_zeros, safe=False)
    Ts = BigMatrix("Ts", shape=(2*N, 2*N, num_tree_levels), shard_sizes=(shard_size, shard_size, 1), write_header=True, parent_fn=constant_zeros, safe=False)
    Rs = BigMatrix("Rs", shape=(2*N, 2*N, num_tree_levels), shard_sizes=(shard_size, shard_size, 1), write_header=True, parent_fn=constant_zeros, safe=False)
    Ss = BigMatrix("Ss", shape=(2*N, 2*N, 2*N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, 1, 1), write_header=True, parent_fn=constant_zeros, safe=False)
    print("Rs", Rs.shape)
    print("Ss", Ss.shape)
    print("Ts", Ts.shape)
    print("Vs", Vs.shape)
    t = time.time()
    p0 = lpcompile_for_execution(QR, inputs=["I"], outputs=["Rs"])
    p1 = p0(A, Vs, Ts, Rs, Ss, N_blocks, 0)
    e = time.time()
    c_time = e - t
    config = npw.config.default()
    program = lp.LambdaPackProgram(p1, config=config)
    return program, {"outputs":[Rs, Vs, Ts], "intermediates":[Ss], "compile_time": c_time}


def bdfac(A, truncate=0):
    b_fac = 2
    N = A.shape[0]
    N_blocks = A.num_blocks(0)
    b_fac = 2
    shard_size = A.shard_sizes[0]
    num_tree_levels = max(int(np.ceil(np.log2(A.num_blocks(0))/np.log2(b_fac))), 1) + 1
    V_QR = BigMatrix("V_QR", shape=(2*N, num_tree_levels, 2*N), shard_sizes=(1, 1, shard_size), write_header=True, safe=False)
    T_QR = BigMatrix("T_QR", shape=(2*N, num_tree_levels, 2*N), shard_sizes=(1, 1, shard_size), write_header=True, safe=False)
    R_QR = BigMatrix("R_QR", shape=(2*N, num_tree_levels, 2*N), parent_fn=constant_zeros, shard_sizes=(shard_size, 1, shard_size), write_header=True, safe=False)
    S_QR = BigMatrix("S_QR", shape=(2*N, num_tree_levels, 2*N, 2*N), parent_fn=constant_zeros, shard_sizes=(1, 1, shard_size, shard_size), write_header=True, safe=False)
    V_LQ = BigMatrix("V_LQ", shape=(2*N, num_tree_levels, 2*N), shard_sizes=(1, 1, shard_size), write_header=True, safe=False)
    T_LQ = BigMatrix("T_LQ", shape=(2*N, num_tree_levels, 2*N), shard_sizes=(1, 1, shard_size), write_header=True, safe=False)
    L_LQ = BigMatrix("L_LQ", shape=(2*N, num_tree_levels, 2*N), parent_fn=constant_zeros_ext, shard_sizes=(1, 1, shard_size), write_header=True, safe=False)
    S_LQ = BigMatrix("S_LQ", shape=(2*N, num_tree_levels, 2*N, 2*N), parent_fn=constant_zeros_ext, shard_sizes=(1, 1, shard_size, shard_size), write_header=True, safe=False)
    t = time.time()
    p0 = lpcompile_for_execution(BDFAC, inputs=["I"], outputs=["R_QR", "L_LQ"])
    p1 = p0(A, V_QR, T_QR, S_QR, R_QR, V_LQ, T_LQ, S_LQ, L_LQ, N_blocks, truncate)
    e = time.time()
    c_time = e - t
    config = npw.config.default()
    program = lp.LambdaPackProgram(p1, config=config)
    return program, {"outputs":[L_LQ, R_QR], "intermediates":[S_LQ, S_QR, T_QR, V_QR, V_LQ, T_LQ], "compile_time": c_time}










