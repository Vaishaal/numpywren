from numpywren.matrix import BigMatrix, BigSymmetricMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
from numpy.linalg import cholesky
import pywren
import unittest

class CholeskyTestClass(unittest.TestCase):
    def test_single_shard_cholesky(self):
        X = np.random.randn(4,4)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        A_sharded= BigMatrix("cholesky_test_A", shape=A.shape, shard_sizes=A.shape)
        y_sharded = BigMatrix("cholesky_test_y", shape=y.shape, shard_sizes=y.shape)
        shard_matrix(A_sharded, A)
        shard_matrix(y_sharded, y)
        L_sharded = binops.chol(pwex, A_sharded, y_sharded)
        L_sharded_local = L_sharded.numpy()
        L = cholesky(A)
        assert(np.allclose(L,L_sharded_local))

    def test_multiple_shard_cholesky_2_2(self):
        np.random.seed(0)
        print("HI")
        X = np.random.randn(128,128)
        A = X.dot(X.T) + np.eye(X.shape[0])
        y = np.random.randn(16)
        pwex = pywren.default_executor()
        shard_sizes = tuple(map(int, np.array(X.shape)/16))
        A_sharded= BigSymmetricMatrix("cholesky_test_A", shape=A.shape, shard_sizes=shard_sizes)
        y_sharded = BigMatrix("cholesky_test_y", shape=y.shape, shard_sizes=shard_sizes[:1])
        print("HIPOTATO")
        shard_matrix(A_sharded, A)
        shard_matrix(y_sharded, y)
        L = cholesky(A)
        print(L)
        L_sharded = binops.chol(pwex, A_sharded, y_sharded)
        L_sharded_local = L_sharded.numpy()
        print(L_sharded_local)
        print(L)
        assert(np.allclose(L,L_sharded_local))


