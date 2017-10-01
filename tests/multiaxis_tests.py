import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, ops
import pytest
import numpy as np
import pywren
import unittest

class MultiAxisTestClass(unittest.TestCase):
    def test_single_multiaxis(self):
        np.random.seed(0)
        X = np.random.randn(8, 8, 8, 8)
        X_sharded = BigMatrix("multiaxis", shape=X.shape, shard_sizes=X.shape)
        X_sharded.shard_matrix(X)
        X_sharded_local = X_sharded.get_block(0,0,0,0)
        X_sharded.free()
        assert(np.all(X_sharded_local == X))



