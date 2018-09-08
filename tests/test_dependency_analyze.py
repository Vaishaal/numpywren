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

def test_simple_linear():
    A = dummy_matrix()
    B = dummy_matrix()
    program = lpcompile(SimpleTestLinear)(A,B,int(3))
    states = walk_program(program)
    for p_idx, loop_vars in states:

        if (p_idx == 0):
            assert (loop_vars['i'] <= loop_vars['j'])
        r_call_abstract_with_scope = program[p_idx]
        scope = r_call_abstract_with_scope.scope.copy()
        children = find_children(program[p_idx], program, **loop_vars)
        print("Current Node", p_idx, loop_vars)
        print("children", children)
        for p_idx, scope in children:
            parents_for_children = find_parents(program[p_idx], program, **scope)
            print(parents_for_children)
            print((p_idx, loop_vars))

            print((p_idx, loop_vars) in parents_for_children)



        print("==========")















if __name__ == "__main__":
    test_simple_linear()

'''
size = 64
shard_size = 16
N = 64
shard_size = 16
shard_sizes = (shard_size, shard_size)
X = np.random.randn(size, size)
X_sharded= BigMatrix("tsqr_test_X", shape=X.shape, shard_sizes=shard_sizes, write_header=False)
b_fac = 2
async def parent_fn(self, loop, *block_idxs):
    if (block_idxs[-1] == 0 and block_idxs[-2] == 0):
        return await X_sharded.get_block_async(None, *block_idxs[:-2])
num_tree_levels = max(int(np.ceil(np.log2(X_sharded.num_blocks(0))/np.log2(b_fac))), 1)
R_sharded= BigMatrix("tsqr_test_R", shape=(num_tree_levels*shard_size, X_sharded.shape[0]), shard_sizes=shard_sizes, write_header=False, safe=False)
V_sharded= BigMatrix("tsqr_test_V", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=False, safe=False)
T_sharded= BigMatrix("tsqr_test_T", shape=(num_tree_levels*shard_size*b_fac, X_sharded.shape[0]), shard_sizes=(shard_size*b_fac, shard_size), write_header=False, safe=False)
I = BigMatrix("I", shape=(N, N), shard_sizes=(shard_size, shard_size), write_header=True, safe=False)
Vs = BigMatrix("Vs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Ts = BigMatrix("Ts", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Rs = BigMatrix("Rs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Ss = BigMatrix("Ss", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True, parent_fn=parent_fn, safe=False)
#tsqr = frontend.lpcompile(TSQR_BinTree)
N_blocks = X_sharded.num_blocks(0)
program_compiled_linear = compiler.lpcompile(SimpleTestLinear)(Vs, Ts)
program_compiled_nonlinear = compiler.lpcompile(SimpleTestNonLinear)(Vs, Ts, 100)
program_compiled_QR = compiler.lpcompile(QR)(I, Vs, Ts, Rs, Ss, 16, 0)

start = timer()
#print("children linear", compiler.find_children(program_compiled_linear[0], program_compiled_linear, level=0, j=4, i=3))
#print("children nonlinear", compiler.find_children(program_compiled_nonlinear[0], program_compiled_nonlinear, level=1, k=8, i=0))
#print("children qr", compiler.find_children(program_compiled_QR[1], program_compiled_QR, i=0, j=0, level=0))
print("children", compiler.find_children(program_compiled_QR[9], program_compiled_QR, k=2, j=3, i=1))
print("parents", compiler.find_parents(program_compiled_QR[6], program_compiled_QR, i=2, j=3))
#print("parents", compiler.find_parents(program_compiled_QR[2], program_compiled_QR, i=0, j=0, level=2))
end = timer()
print(end - start)
'''
