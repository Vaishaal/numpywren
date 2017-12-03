
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import local_numpy_init
import pytest
import numpy as np
import pywren
import unittest
import time
import os

class FastIOTestClass(unittest.TestCase):
    def test_sharded_matrix_row_get(self):
        X = np.random.randn(8,8)
        X_sharded = local_numpy_init(X, shard_sizes=[1, 1])
        row_0 = matrix_utils.get_row(X_sharded, 0)
        X_sharded.free()
        print(X[0].shape)
        print(row_0.shape)
        os.system("rm -rf /dev/shm/*")
        assert(np.all(X[0] == row_0))

    def test_sharded_matrix_row_get_big(self):
        s = 2
        X = np.arange(0,2048*2048*s).reshape(2048, 2048*s)
        X_sharded = local_numpy_init(X, shard_sizes=[2048, 2048])
        t = time.time()
        row_0 = matrix_utils.get_row(X_sharded, 0)
        e = time.time()
        print(row_0.shape)
        print("Effective GB/s", (2048*2048*s*8)/ (1e9*(e - t)))
        print("Download Time", e - t)
        X_sharded.free()
        os.system("rm -rf /dev/shm/*")
        assert(np.all(X == row_0))

    def test_sharded_matrix_row_put_big(self):
        s = 2
        X = np.arange(0,2048*2048*s).reshape(2048, 2048*s)
        X_sharded = BigMatrix("row_put_test", shape=X.shape, shard_sizes=[2048, 2048])
        t = time.time()
        matrix_utils.put_row(X_sharded, X, 0)
        e = time.time()
        print(X.shape)
        print("Effective GB/s", (2048*2048*s*8)/ (1e9*(e - t)))
        print("Upload Time", e - t)

        t = time.time()
        row_0 = matrix_utils.get_row(X_sharded, 0)
        e = time.time()
        X_sharded.free()
        os.system("rm -rf /dev/shm/*")
        assert(np.all(X == row_0))


