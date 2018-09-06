from numpywren import frontend
from numpywren.matrix import BigMatrix
import numpy as np
from numpywren.matrix_init import shard_matrix
import time
from timeit import default_timer as timer

def SimpleTestLinear(A:BigMatrix):
    for i in range(100):
        for j in range(i, 100):
            A[j, i] = identity(A[i,j])

def SimpleTestNonLinear(A:BigMatrix, B: BigMatrix, N:int):
    for i in range(N):
        N_tree = ceiling(log(N - i))/log(2)
        for level in range(0,ceiling(log(N - i))/log(2)):
            for k in range(i, N, 2**(level+1)):
                A[N_tree - level - 1, i, k] = add(A[N_tree - level , i, k], A[N_tree - level, i, k + 2**(level)])

        B[i] = identity(A[1, i, 0])




'''


def TSQR_BinTree(A:BigMatrix, Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, N:int):
    for level in range(0, ceiling(log(N)/log(2))):
        for j in range(0, N, 2**(level + 1)):
            Vs[level+1, j], Ts[level+1, j], Rs[level+1, j] = qr_factor(Rs[level, j], Rs[level, j + 2**(level)])


def QR(Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, S:BigMatrix, N:int, truncate:int):
    b_fac = 2
    for i in range(0, N):
        N_tree = ceiling(log(N - i)/log(2))
        for j in range(i, N):
            Vs[j, i, N_tree], Ts[j, i, N_tree], Rs[j, i, N_tree] = qr_factor(S[j, i, i, 0])

        for j in range(0, N, 2):
            Vs[j, i, N_tree - 1], Ts[j, i, N_tree - 1], Rs[j, i, N_tree - 1] = qr_factor(Rs[j, i, N_tree], Rs[j + 2, i, N_tree])

        for level in range(1, N_tree):
            for j in range(0, N, 2**(level + 1)):
                Vs[j, i, N_tree - level - 1], Ts[j, i, N_tree - level - 1], Rs[j, i, N_tree - level - 1] = qr_factor(Rs[j, i, N_tree - level], Rs[j + 2**(level), i, N_tree - level])

        # flat trailing matrix update
        for j in range(i, N):
            for k in range(i+1, N):
                S[j, k, i+1, N_tree] = qr_leaf(Vs[j, i, N_tree], Ts[j, i, N_tree], S[j, k, i, 0])

        for k in range(i+1, N):
            for j in range(0, N, 2):
                S[j, k, i+1, N_tree - 1], S[j + 2, k, i+1, 0]  = qr_trailing_update(Vs[j, i, N_tree - 1], Ts[j, i, N_tree - 1], S[j, k, i+1, N_tree], S[j + 2, k, i +1, N_tree])

            for level in range(1, N_tree):
                for j in range(0, N, 2**(level + 1)):
                    S[j, k, i+1, N_tree - 1 - level], S[j + 2**level, k, i+1, 0]  = qr_trailing_update(Vs[j, i, N_tree - 1 - level], Ts[j, i, N_tree - 1 - level], S[j, k, i+1, N_tree - level], S[j + 2**level, k, i +1, N_tree - level])

        for k in range(i+1, N):
            Rs[i, k, 0]  = identity(S[i, k, i+1, 0])
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
Vs = BigMatrix("Vs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Ts = BigMatrix("Ts", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Rs = BigMatrix("Rs", shape=(num_tree_levels, N, N), shard_sizes=(1, shard_size, shard_size), write_header=True, safe=False)
Ss = BigMatrix("Ss", shape=(N, N, N, num_tree_levels*shard_size), shard_sizes=(shard_size, shard_size, shard_size, shard_size), write_header=True, parent_fn=parent_fn, safe=False)
#tsqr = frontend.lpcompile(TSQR_BinTree)
N_blocks = X_sharded.num_blocks(0)
program_compiled = frontend.lpcompile(SimpleTestLinear)(Vs)

#print(pc)
#print(frontend.eval_remote_call(program_compiled[0], level=0, j=0))
start = timer()
print("children ", frontend.find_children(program_compiled[0], program_compiled, level=0, j=0, i=0))
end = timer()
print(f"{end - start}")
