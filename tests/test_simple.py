import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
import pytest
import numpy as np
import pywren import unittest


class SimpleTestClass(unittest.TestCase):
    def test_single_shard_matrix(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        X_sharded = BigMatrix("test_0", shape=X.shape, shard_sizes=X.shape)
        X_sharded.shard_matrix(X)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X_sharded_local == X))

    def test_multiple_shard_matrix(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("test_1", shape=X.shape, shard_sizes=shard_sizes)
        X_sharded.shard_matrix(X)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X == X_sharded_local))


if __name__ == "__main__":
    tests = SimpleTestClass()
    tests.test_single_shard_matrix()
    tests.test_multiple_shard_matrix()


