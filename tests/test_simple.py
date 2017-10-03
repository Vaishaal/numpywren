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


if __name__ == "__main__":
    tests = SimpleTestClass()
    tests.test_single_shard_matrix()
    tests.test_multiple_shard_matrix()


