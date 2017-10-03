import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
import pywren
import unittest

class GemvTestClass(unittest.TestCase):
    def test_single_shard_gemv(self):
        X = np.random.randn(16,16)
        Y = np.random.randn(16)
        X_sharded = BigMatrix("gemv_test_0", shape=X.shape, shard_sizes=X.shape)
        Y_sharded = BigMatrix("gemv_test_2", shape=Y.shape, shard_sizes=Y.shape)
        shard_matrix(X_sharded, X)
        pwex = pywren.default_executor()
        XY_sharded = binops.gemv(pwex, X_sharded, Y_sharded, X_sharded.bucket, 1)
        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        print(XY)
        print(XY_sharded_local)
        X_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY,XY_sharded_local)))

    def test_multiple_shard_gemv(self):
        X = np.random.randn(16,16)
        Y = np.random.randn(16)
        shard_sizes_0 = tuple(map(int, np.array(X.shape)/2))
        shard_sizes_1 = tuple(map(int, np.array(Y.shape)/2))
        X_sharded = BigMatrix("gemv_test_1", shape=X.shape, shard_sizes=shard_sizes_0)
        Y_sharded = BigMatrix("gemv_test_2", shape=Y.shape, shard_sizes=shard_sizes_1)
        shard_matrix(X_sharded, X)
        shard_matrix(Y_sharded, Y)
        pwex = pywren.default_executor()
        XY_sharded = binops.gemv(pwex, X_sharded, Y_sharded, X_sharded.bucket, 1)
        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        X_sharded.free()
        Y_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY,XY_sharded_local)))

    def test_multiple_shard_matrix_gemv(self):
        X = np.random.randn(16,16)
        Y = np.random.randn(16,1)
        shard_sizes_0 = tuple(map(int, np.array(X.shape)/2))
        shard_sizes_1 = (Y.shape[0], 1)
        X_sharded = BigMatrix("gemv_test_1", shape=X.shape, shard_sizes=shard_sizes_0)
        Y_sharded = BigMatrix("gemv_test_2", shape=Y.shape, shard_sizes=shard_sizes_1)
        shard_matrix(X_sharded, X)
        shard_matrix(Y_sharded, Y)
        pwex = pywren.default_executor()
        XY_sharded = binops.gemv(pwex, X_sharded, Y_sharded, X_sharded.bucket, 1)
        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        X_sharded.free()
        Y_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY,XY_sharded_local)))
