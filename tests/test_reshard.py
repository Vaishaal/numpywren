import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops, matrix_init
from numpywren.matrix_init import local_numpy_init
import pytest
import numpy as np
import pywren
import unittest


class ReshardTest(unittest.TestCase):
    def test_single_shard_matrix(self):
        pwex = pywren.default_executor()
        X = np.random.randn(128,128)
        X_sharded = local_numpy_init(X, (128,128))
        X_sharded_down = matrix_init.reshard_down(X_sharded, (4,4), pwex)
        X_sharded_local = X_sharded.numpy()
        X_sharded_local_2 = X_sharded_down.numpy()
        X_sharded.free()
        X_sharded_down.free()
        assert(np.all(X_sharded_local == X))
        assert(np.all(X_sharded_local_2 == X))

    def test_multiple_shard_matrix(self):
        pwex = pywren.default_executor()
        X = np.random.randn(128,128)
        X_sharded = local_numpy_init(X, (64,64))
        X_sharded_down = matrix_init.reshard_down(X_sharded, (4,4), pwex)
        X_sharded_local = X_sharded.numpy()
        X_sharded_local_2 = X_sharded_down.numpy()
        X_sharded.free()
        X_sharded_down.free()
        assert(np.all(X_sharded_local == X))
        assert(np.all(X_sharded_local_2 == X))

    def test_multiple_shard_tensor(self):
        pwex = pywren.default_executor()
        X = np.random.randn(128,128, 4)
        X_sharded = local_numpy_init(X, (64,64, 4))
        X_sharded.autosqueeze = False
        X_sharded_down = matrix_init.reshard_down(X_sharded, (4,4,2), pwex=None)
        X_sharded_local = X_sharded.numpy()
        X_sharded_local_2 = X_sharded_down.numpy()
        X_sharded.free()
        X_sharded_down.free()
        X_sharded_down.get_block(0,0,0)
        assert(np.all(X_sharded_local == X))
        assert(np.all(X_sharded_local_2 == X))






if __name__ == "__main__":
    tests = SimpleTestClass()
    tests.test_single_shard_matrix()
    tests.test_multiple_shard_matrix()


