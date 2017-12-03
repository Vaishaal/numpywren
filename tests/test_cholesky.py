from numpywren.matrix import BigMatrix, BigSymmetricMatrix
from numpywren import matrix_utils, uops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest
import concurrent.futures as fs
import time
import multiprocessing
import os
cpu_count = multiprocessing.cpu_count()



class CholeskyTestClass(unittest.TestCase):
    def test_single_shard_cholesky(self):
        X = np.random.randn(4,4)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=A.shape)
        y_sharded = BigMatrix("cholesky_test_y", shape=y.shape, shard_sizes=y.shape)
        shard_matrix(A_sharded, A)
        shard_matrix(y_sharded, y)
        L_sharded = uops.chol(pwex, A_sharded)
        L_sharded_local = L_sharded.numpy()
        L = cholesky(A)
        assert(np.allclose(L,L_sharded_local))
        os.system("rm -rf /dev/shm/*")

    def test_multiple_shard_cholesky(self):
        np.random.seed(1)
        size = 128
        shard_size = 64
        np.random.seed(1)
        print("Generating X")
        executor = fs.ProcessPoolExecutor(cpu_count)
        X = np.random.randn(size, 128)
        print("Generating A")
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(size)
        pwex = pywren.default_executor()
        print("sharding A")
        shard_sizes = (shard_size, shard_size)
        A_sharded= BigSymmetricMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes)
        y_sharded = BigMatrix("cholesky_test_y", shape=y.shape, shard_sizes=shard_sizes[:1])
        A_sharded.free()
        y_sharded.free()
        A_sharded= BigSymmetricMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes)
        y_sharded = BigMatrix("cholesky_test_y", shape=y.shape, shard_sizes=shard_sizes[:1])
        t = time.time()
        shard_matrix(A_sharded, A, executor=executor)
        e = time.time()
        print("A_sharded", e - t)
        t = time.time()
        shard_matrix(y_sharded, y, executor=executor)
        e = time.time()
        print("y_sharded time", e - t)
        print("Computing LL^{T}")
        L = cholesky(A)
        print(L)
        L_sharded = uops.chol(pwex, A_sharded)
        L_sharded_local = L_sharded.numpy()
        print(L_sharded_local)
        print(L)
        print("L_{infty} difference ", np.max(np.abs(L_sharded_local - L)))
        assert(np.allclose(L,L_sharded_local))
        os.system("rm -rf /dev/shm/*")

if __name__ == "__main__":
    tests = CholeskyTestClass()
    tests.test_multiple_shard_cholesky()
