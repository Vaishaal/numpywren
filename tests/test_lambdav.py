import sklearn.datasets as datasets
from numpywren.matrix import BigSymmetricMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import shard_matrix
import numpy as np
import pytest
import pywren
import unittest
import os

class LambdavTest(unittest.TestCase):
    def test_lambdav(self):
        X = np.random.randn(18,18)
        X = X.dot(X.T)
        X_sharded = BigSymmetricMatrix("lambdav", shape=X.shape, shard_sizes=[4,4], lambdav=7.0)
        shard_matrix(X_sharded, X)
        X += 7.0*np.eye(X.shape[0])
        X_local = X_sharded.numpy()
        X_sharded.free()
        assert(np.all(np.isclose(X,X_local)))
        os.system("rm -rf /dev/shm/*")
