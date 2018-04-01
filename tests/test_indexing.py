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
        X_sharded_local = X_sharded.submatrix(0, 0).get_block()
        assert(np.all(X_sharded_local == X))

    def test_single_shard_index_put(self):
        X = np.random.randn(128,128)
        X_sharded = BigMatrix("test_1", shape=X.shape, shard_sizes=X.shape)
        X_sharded.submatrix(0, 0).put_block(X)
        assert(np.all(X_sharded.numpy() == X))

    def test_multiple_shard_index_get(self):
        X = np.random.randn(128,128)
        shard_sizes = [64, 64] 
        X_sharded = BigMatrix("test_2", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        assert(np.all(X[0:64, 0:64] == X_sharded.submatrix(0).get_block(0)))
        assert(np.all(X[64:128, 64:128] == X_sharded.submatrix(1, 1).get_block()))
        assert(np.all(X[0:64, 64:128] == X_sharded.submatrix(0, 1).get_block()))
        assert(np.all(X[64:128, 0:64] == X_sharded.submatrix(None, 0).get_block(1)))

    def test_simple_slices(self):
        X = np.random.randn(128,128)
        shard_sizes = [32, 32] 
        X_sharded = BigMatrix("test_3", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        assert(np.all(X[0:64] == X_sharded.submatrix([2]).numpy()))
        assert(np.all(X[64:128] == X_sharded.submatrix([2, None]).numpy()))
        assert(np.all(X[:, 0:96] == X_sharded.submatrix(None, [0, 3]).numpy()))
        assert(np.all(X[:, 96:128] == X_sharded.submatrix(None, [3, None]).numpy()))

    def test_step_slices(self):
        X = np.random.randn(128,128)
        shard_sizes = [16, 16] 
        X_sharded = BigMatrix("test_4", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        assert(np.all(X[::32] == X_sharded.submatrix([None, None, 2]).numpy()[::16]))
        assert(np.all(X[16::32] == X_sharded.submatrix([1, None, 2]).numpy()[::16]))
        assert(np.all(X[:, 0:96:64] == X_sharded.submatrix(None, [0, 6, 4]).numpy()[:, ::16]))
        assert(np.all(X[:, 96:128:64] == X_sharded.submatrix(None, [6, 8, 4]).numpy()[:, ::16]))
    """
    def test_complex_slices(self):
        X = np.random.randn(21, 67, 53, 27)
        shard_sizes = [3, 16, 11, 9] 
        X_sharded = BigMatrix("test_5", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        assert(np.all(X[::32] == X_sharded.submatrix([None, None, 2]).numpy()[::16]))
        assert(np.all(X[16::32] == X_sharded.submatrix([1, None, 2]).numpy()[::16]))
        assert(np.all(X[:, 0:96:64] == X_sharded.submatrix(None, [0, 3, 2]).numpy()[:, ::16]))
        assert(np.all(X[:, 96:128:64] == X_sharded.submatrix(None, [3, None]).numpy()[:, ::16]))
    """
