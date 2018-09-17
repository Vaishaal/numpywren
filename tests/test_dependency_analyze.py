import time
import random
from timeit import default_timer as timer
import string

from numpywren import compiler
from numpywren.matrix import BigMatrix
import numpy as np
from numpywren.matrix_init import shard_matrix
from numpywren.algs import *
from numpywren.compiler import lpcompile, walk_program, find_parents, find_children

def dummy_matrix(key_len=256, num_dims=2):
    key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(key_len))
    shape = tuple([1 for x in range(num_dims)])
    return BigMatrix(key, shape=shape, shard_sizes=shape, write_header=False)

def verify_program(program):
    states = walk_program(program)
    for p_idx, loop_vars in states:
        r_call_abstract_with_scope = program[p_idx]
        current_node = (p_idx, loop_vars)
        scope = r_call_abstract_with_scope.scope.copy()
        children = find_children(program, p_idx, loop_vars)
        for p_idx_child, child_vars in children:
            child_parents = find_parents(program, p_idx_child, child_vars)
            assert current_node in child_parents
        parents = find_parents(program, p_idx, loop_vars)
        for p_idx_parent, parent_vars in parents:
            parent_children = find_children(program, p_idx_parent, parent_vars)
            assert current_node in parent_children

def test_simple_linear():
    A = dummy_matrix()
    B = dummy_matrix()
    program = lpcompile(SimpleTestLinear)(A,B,int(8))
    verify_program(program)

def test_simple_linear_2():
    A = dummy_matrix()
    B = dummy_matrix()
    program = lpcompile(SimpleTestLinear2)(A,B,int(8))
    verify_program(program)

def test_simple_nonlinear():
    A = dummy_matrix()
    B = dummy_matrix()
    program = lpcompile(SimpleTestNonLinear)(A,B,int(8))
    verify_program(program)

def test_cholesky():
    A = dummy_matrix()
    B = dummy_matrix()
    C = dummy_matrix(num_dims=3)
    program = lpcompile(CHOLESKY)(A,B,C,8,0)
    verify_program(program)

def test_tsqr():
    A = dummy_matrix(num_dims=1)
    V = dummy_matrix(num_dims=2)
    T = dummy_matrix(num_dims=2)
    R = dummy_matrix(num_dims=2)
    program = lpcompile(TSQR)(A,V,T,R,16)
    verify_program(program)

def test_qr():
    A = dummy_matrix(num_dims=2)
    V = dummy_matrix(num_dims=2)
    T = dummy_matrix(num_dims=2)
    R = dummy_matrix(num_dims=2)
    S = dummy_matrix(num_dims=4)
    t = time.time()
    program = lpcompile(QR)(A,V,T,R,S,16,0)
    #states = walk_program(program)
    #idx = np.random.choice(len(states),  len(states) - 1, replace=False)[0]
    #states = states[idx:idx+1]
    states = [(8, {'j': 1, 'level': 0, 'k': 6, 'i': 1})]
    print(states)
    times = []
    print("number of states", len(states))
    actual = time.time()
    for p_idx, loop_vars in states:
        r_call_abstract_with_scope = program[p_idx]
        current_node = (p_idx, loop_vars)
        scope = r_call_abstract_with_scope.scope.copy()
        children = find_children(program, p_idx, loop_vars)
        for p_idx_child, child_vars in children:
            child_parents = find_parents(program, p_idx_child, child_vars)
            assert current_node in child_parents
    e = time.time()
    times.append(e - actual)
    print("mean find children and parents", np.mean(times))
    print("full time ", e - t)
    print("std time find children and parents", np.std(times))
    #verify_program(program)

def test_gemm():
    A = dummy_matrix(num_dims=2)
    B = dummy_matrix(num_dims=2)
    M = 4
    N = 4
    K = 4
    Temp = dummy_matrix(num_dims=4)
    Out = dummy_matrix(num_dims=3)
    program = lpcompile(GEMM)(A,B,M,N,K,Temp,Out)
    verify_program(program)

def test_bdfac():
    A = dummy_matrix(num_dims=2)
    V = dummy_matrix(num_dims=3)
    T = dummy_matrix(num_dims=3)
    R = dummy_matrix(num_dims=4)
    S = dummy_matrix(num_dims=4)
    VR = dummy_matrix(num_dims=3)
    TR = dummy_matrix(num_dims=3)
    L = dummy_matrix(num_dims=4)
    SR = dummy_matrix(num_dims=4)
    program = lpcompile(BDFAC)(A,V,T,R,S,VR,TR,SR,L,256,0)
    t = time.time()
    print(find_children(program, 15, {'j': 2, 'k': 2, 'i': 1, 'level': 1}))
    e = time.time()
    print(e - t)
    #verify_program(program)



if __name__ == "__main__":
    test_bdfac()

