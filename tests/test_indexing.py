import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
import pywren
import unittest

class IndexingTestClass(unittest.TestCase):
    def test_single_shard_index_get(self):
        X = np.random.randn(128,128)
        X_sharded = BigMatrix("test_0", shape=X.shape, shard_sizes=X.shape)
        shard_matrix(X_sharded, X)
        X_sharded_local = X_sharded[0, 0].get_block()
        assert(np.all(X_sharded_local == X))

    def test_single_shard_index_put(self):
        X = np.random.randn(128,128)
        X_sharded = BigMatrix("test_1", shape=X.shape, shard_sizes=X.shape)
        X_sharded[0, 0].put_block(X)
        assert(np.all(X_sharded.numpy() == X))

    def test_multiple_shard_index_get(self):
        X = np.random.randn(128,128)
        shard_sizes = [64, 64] 
        X_sharded = BigMatrix("test_2", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        X_sharded_local = np.empty([128, 128])
        assert(np.all(X[0:64, 0:64] == X_sharded[0].get_block(0)))
        assert(np.all(X[64:128, 64:128] == X_sharded[1, 1].get_block()))
        assert(np.all(X[0:64, 64:128] == X_sharded[0, 1].get_block()))
        assert(np.all(X[64:128, 0:64] == X_sharded[:, 0].get_block(1)))
