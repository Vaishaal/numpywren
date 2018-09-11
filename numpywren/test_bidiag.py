from kernels import banded_to_bidiagonal
import numpy as np
import scipy.linalg

l = []
num_blocks = 3
shard_size = 2 
for i in range(2 * (num_blocks - 1) + 1):
    if i % 2 == 0:
        l.append(np.triu(np.random.randn(shard_size, shard_size)))
    else:
        l.append(np.tril(np.random.randn(shard_size, shard_size)))
mat = [[np.zeros([shard_size, shard_size]) for i in range(num_blocks)] for j in range(num_blocks)]

mat[0][0] = l[0]
for i in range(1, num_blocks):
    mat[i][i] = l[2 * i]
    mat[i - 1][i] = l[2 * i - 1]
mat = np.block(mat)

s1 = scipy.linalg.svd(mat, compute_uv=False)
print(mat)


diag, offdiag = banded_to_bidiagonal(l)
res = np.diag(diag) + np.diag(offdiag, k=1)
s2 = scipy.linalg.svd(res, compute_uv=False)

print(s1)
print(s2)
