import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
import pywren
import unittest

class MultiAxisTestClass(unittest.TestCase):
    def test_single_multiaxis(self):
        X = np.random.randn(8, 8, 8, 8)
        X_sharded = BigMatrix("multiaxis", shape=X.shape, shard_sizes=X.shape)
        print("BLOCK_IDXS", X_sharded.block_idxs)
        shard_matrix(X_sharded, X)
        print("BLOCK_IDXS_EXIST", X_sharded.block_idxs_exist)
        X_sharded_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(X_sharded_local == X))

    def test_sharded_multiaxis(self):
        X = np.random.randn(8, 8, 8, 8)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("multiaxis_2", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        print("BLOCK_IDXS", X_sharded.block_idxs)
        X_sharded_local = X_sharded.numpy()
        print(X_sharded.free())
        assert(np.all(X_sharded_local == X))





