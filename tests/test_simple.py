import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import local_numpy_init
import pytest
import numpy as np
import pywren
import unittest


class SimpleTestClass(unittest.TestCase):
    def test_single_shard_matrix(self):
        X = np.random.randn(128,128)
        X_sharded = local_numpy_init(X, X.shape)
        local_numpy_init(X, shard_sizes=X.shape)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X_sharded_local == X))

    def test_multiple_shard_matrix(self):
        X = np.random.randn(128,128)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = local_numpy_init(X, shard_sizes=shard_sizes)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X == X_sharded_local))

    def test_matrix_header(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = local_numpy_init(X, shard_sizes=shard_sizes, write_header=True)
        X_sharded_local = X_sharded.numpy()
        assert(np.all(X == X_sharded_local))
        X_sharded_2 = BigMatrix(X_sharded.key)
        X_sharded_local_2 = X_sharded_2.numpy()
        assert(np.all(X == X_sharded_local_2))

    def test_multiple_shard_symmetric_matrix(self):
        X = np.random.randn(128,128)
        X = X.dot(X.T)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = local_numpy_init(X, shard_sizes=shard_sizes, symmetric=True)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X == X_sharded_local))


    def test_multiple_shard_matrix_uneven(self):
        X = np.random.randn(200,200)
        shard_sizes = (101,101)
        X_sharded = local_numpy_init(X, shard_sizes=shard_sizes)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X == X_sharded_local))

    def test_multiple_shard_symmetric_matrix_uneven(self):
        X = np.random.randn(200,200)
        shard_sizes = (101,101)
        X = X.T.dot(X)
        X_sharded = local_numpy_init(X, shard_sizes=shard_sizes, symmetric=True)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X == X_sharded_local))





if __name__ == "__main__":
    tests = SimpleTestClass()
    tests.test_single_shard_matrix()
    tests.test_multiple_shard_matrix()


