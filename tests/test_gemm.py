import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import numpy as np
import pytest
import pywren
import unittest
import os

class GemmTestClass(unittest.TestCase):
    def test_single_shard_matrix_multiply(self):
        X = np.random.randn(16,16)
        X_sharded = BigMatrix("gemm_test_0", shape=X.shape, shard_sizes=X.shape)
        shard_matrix(X_sharded, X)
        pwex = pywren.default_executor()
        XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, X_sharded.bucket, 1)
        XXT_sharded_local = XXT_sharded.numpy()
        XXT = X.dot(X.T)
        print(XXT)
        print(XXT_sharded_local)
        X_sharded.free()
        XXT_sharded.free()
        assert(np.all(np.isclose(XXT,XXT_sharded_local)))
        os.system("rm -rf /dev/shm/*")

    def test_multiple_shard_matrix_multiply_symmetric(self):
        X = np.random.randn(16,16)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        pwex = pywren.default_executor()
        XXT_sharded = binops.gemm(pwex, X_sharded, X_sharded.T, X_sharded.bucket, 1)
        XXT_sharded_local = XXT_sharded.numpy()
        XXT = X.dot(X.T)
        print(XXT)
        print(XXT_sharded_local)
        X_sharded.free()
        XXT_sharded.free()
        assert(np.all(np.isclose(XXT,XXT_sharded_local)))
        os.system("rm -rf /dev/shm/*")

    def test_multiple_shard_matrix_multiply_symmetric_2(self):
        X = np.random.randn(16,16)
        shard_sizes = [8,16]
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        pwex = pywren.default_executor()
        XTX_sharded = binops.gemm(pwex, X_sharded.T, X_sharded, X_sharded.bucket, 1, local=True)
        XTX_sharded_local = XTX_sharded.numpy()
        XTX = X.T.dot(X)
        X_sharded.free()
        XTX_sharded.free()
        print(np.linalg.norm(XTX - XTX_sharded_local))
        assert(np.all(np.isclose(XTX,XTX_sharded_local)))
        os.system("rm -rf /dev/shm/*")


    def test_multiple_shard_matrix_multiply(self):
        X = np.random.randn(16,16)
        Y = np.random.randn(16,16)
        shard_sizes = tuple(map(int, np.array(X.shape)/2))
        X_sharded = BigMatrix("gemm_test_1", shape=X.shape, shard_sizes=shard_sizes)
        Y_sharded = BigMatrix("gemm_test_2", shape=X.shape, shard_sizes=shard_sizes)
        shard_matrix(X_sharded, X)
        shard_matrix(Y_sharded, Y)
        pwex = pywren.default_executor()
        XY_sharded = binops.gemm(pwex, X_sharded, Y_sharded, X_sharded.bucket, 1)
        XY_sharded_local = XY_sharded.numpy()
        XY = X.dot(Y)
        X_sharded.free()
        Y_sharded.free()
        XY_sharded.free()
        assert(np.all(np.isclose(XY,XY_sharded_local)))
        os.system("rm -rf /dev/shm/*")
