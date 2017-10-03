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
        X = np.random.randn(2,2)
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

    def test_multiple_shard_matrix_multiply(self):
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
