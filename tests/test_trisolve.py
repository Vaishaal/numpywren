import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import numpy as np
import pytest
import pywren
import unittest
import os
import scipy.linalg
import warnings

class TrisolveTestClass(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)

    def test_single_shard_trisolve_upper(self):
        A = np.triu(np.random.rand(16,16)) + 1
        A_sharded = BigMatrix("A_trisolve_test_0", shape=A.shape, shard_sizes=A.shape)
        shard_matrix(A_sharded, A)
        B = np.random.rand(16, 1) + 1
        B_sharded = BigMatrix("B_trisolve_test_0", shape=B.shape, shard_sizes=B.shape)
        shard_matrix(B_sharded, B)
        pwex = pywren.lambda_executor()
        X_sharded = binops.trisolve(pwex, A_sharded, B_sharded)
        X_sharded_local = X_sharded.numpy()
        X = scipy.linalg.solve_triangular(A, B) 
        X_sharded.free()
        assert(np.all(np.isclose(X,X_sharded_local)))

    def test_single_shard_trisolve_lower(self):
        A = np.tril(np.random.rand(16,16)) + 1
        A_sharded = BigMatrix("A_trisolve_test_1", shape=A.shape, shard_sizes=A.shape)
        shard_matrix(A_sharded, A)
        B = np.random.rand(16, 1) + 1
        B_sharded = BigMatrix("B_trisolve_test_1", shape=B.shape, shard_sizes=B.shape)
        shard_matrix(B_sharded, B)
        pwex = pywren.lambda_executor()
        X_sharded = binops.trisolve(pwex, A_sharded, B_sharded, lower=True)
        X_sharded_local = X_sharded.numpy()
        X = scipy.linalg.solve_triangular(A, B, lower=True) 
        X_sharded.free()
        assert(np.all(np.isclose(X,X_sharded_local)))

    def test_multiple_shard_trisolve(self):
        A = np.triu(np.random.rand(128,128)) + 1
        A_sharded = BigMatrix("A_trisolve_test_2", shape=A.shape, shard_sizes=[32, 32])
        shard_matrix(A_sharded, A)
        B = np.random.rand(128, 64) + 1
        B_sharded = BigMatrix("B_trisolve_test_2", shape=B.shape, shard_sizes=[32,32])
        shard_matrix(B_sharded, B)
        pwex = pywren.lambda_executor()
        X_sharded = binops.trisolve(pwex, A_sharded, B_sharded)
        X_sharded_local = X_sharded.numpy()
        X = scipy.linalg.solve_triangular(A, B) 
        X_sharded.free()
        assert(np.all(np.isclose(X,X_sharded_local)))

    def test_multiple_uneven_shard_trisolve_upper(self):
        A = np.triu(np.random.rand(128,128)) + 1
        A_sharded = BigMatrix("A_trisolve_test_3", shape=A.shape, shard_sizes=[56, 56])
        shard_matrix(A_sharded, A)
        B = np.random.rand(128, 150) + 1
        B_sharded = BigMatrix("B_trisolve_test_3", shape=B.shape, shard_sizes=[56, 51])
        shard_matrix(B_sharded, B)
        pwex = pywren.lambda_executor()
        X_sharded = binops.trisolve(pwex, A_sharded, B_sharded)
        X_sharded_local = X_sharded.numpy()
        X = scipy.linalg.solve_triangular(A, B) 
        X_sharded.free()
        assert(np.all(np.isclose(X,X_sharded_local)))

    def test_multiple_uneven_shard_trisolve_lower(self):
        A = np.tril(np.random.rand(233,233)) + 1
        A_sharded = BigMatrix("A_trisolve_test_4", shape=A.shape, shard_sizes=[101, 101])
        shard_matrix(A_sharded, A)
        B = np.random.rand(233, 301) + 1
        B_sharded = BigMatrix("B_trisolve_test_1", shape=B.shape, shard_sizes=[101, 120])
        shard_matrix(B_sharded, B)
        pwex = pywren.lambda_executor()
        X_sharded = binops.trisolve(pwex, A_sharded, B_sharded, lower=True)
        X_sharded_local = X_sharded.numpy()
        X = scipy.linalg.solve_triangular(A, B, lower=True) 
        X_sharded.free()
        assert(np.all(np.isclose(X,X_sharded_local)))
