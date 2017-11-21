
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, binops
from numpywren.matrix_init import local_numpy_init
import pytest
import numpy as np
import pywren
import unittest

class FastIOTestClass(unittest.TestCase):
    def test_sharded_matrix_row_get(self):
        X = np.random.randn(8,8)
        X_sharded = local_numpy_init(X, shard_sizes=[1, 1])
        row_0 = matrix_utils.get_row(X_sharded, 0)
        X_sharded.free()
        print(X[0].shape)
        print(row_0.shape)
        assert(np.all(X[0] == row_0))

