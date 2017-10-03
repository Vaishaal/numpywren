import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
import pytest
import numpy as np
import pywren
import unittest

class GemmTestClass(unittest.TestCase):
    def test_single_shard_matrix_multiply(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        X_sharded = BigMatrix("gemm_test_0", shape=X.shape, shard_sizes=X.shape)
        X_sharded.shard_matrix(X)
        pwex = pywren.default_executor()
        XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, X_sharded.bucket, 1)
        XXT_sharded_local = XXT_sharded.numpy()
        XXT = X.dot(X.T)
        print(XXT)
        print(XXT_sharded_local)
        X_sharded.free()
        XXT_sharded.free()
        assert(np.all(np.isclose(XXT,XXT_sharded_local)))

    def test_multiple_shard_matrix_multiply_symmetric(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=shard_sizes)
        X_sharded.shard_matrix(X)
        pwex = pywren.default_executor()
        XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, X_sharded.bucket, 1)
        XXT_sharded_local = XXT_sharded.numpy()
        XXT = X.dot(X.T)
        print(XXT)
        print(XXT_sharded_local)
        X_sharded.free()
        XXT_sharded.free()
        assert(np.all(np.isclose(XXT,XXT_sharded_local)))

    def test_multiple_shard_matrix_multiply(self):
        np.random.seed(0)
        X = np.random.randn(128,128)
        Y = np.random.randn(128,128)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=shard_sizes)
        Y_sharded = BigMatrix("gemm_test_2", shape=X.shape, shard_sizes=shard_sizes)
        X_sharded.shard_matrix(X)
        Y_sharded.shard_matrix(Y)
        pwex = pywren.default_executor()
        XY_sharded = binops.gemm(pwex, X_sharded, Y_sharded, X_sharded.bucket, 1)
        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        X_sharded.free()
        Y_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY,XY_sharded_local)))
