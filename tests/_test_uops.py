from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, uops
from numpywren.matrix_init import shard_matrix
import pytest
import numpy as np
import pywren
import unittest

class UopsTestClass(unittest.TestCase):
    def test_axiswise_uop(self, f, f_numpy, axis):
        X = np.random.randn(16,16)
        pwex = pywren.default_executor()
        X_sharded = BigMatrix("{0}_uop_test".format(f), shape=X.shape, shard_sizes=X.shape)
        shard_matrix(X_sharded, X)
        res_sharded = f(pwex, X_sharded, axis=axis)
        res = res_sharded.numpy()
        res.free()
        assert(np.isclose(f_numpy(X, axis=axis), res))

    def test_elemwise_uop(self, f, f_numpy):
        X = np.random.randn(16,16)
        pwex = pywren.default_executor()
        X_sharded = BigMatrix("{0}_uop_test".format(f), shape=X.shape, shard_sizes=X.shape)
        shard_matrix(X_sharded, X)
        res_sharded = f(pwex, X_sharded)
        res = res_sharded.numpy()
        res.free()
        assert(np.isclose(f_numpy(X), res))

    def test_sum(self):
        self.test_elemwise_uop(uops.sum, np.sum)
        self.test_elemwise_uop(uops.sum, np.sum, axis=0)
        self.test_elemwise_uop(uops.sum, np.sum, axis=1)

    def test_prod(self):
        self.test_elemwise_uop(uops.prod, np.prod)
        self.test_elemwise_uop(uops.prod, np.prod, axis=0)
        self.test_elemwise_uop(uops.prod, np.prod, axis=1)

    def test_argmin(self):
        self.test_elemwise_uop(uops.argmin, np.argmin)
        self.test_elemwise_uop(uops.argmin, np.argmin, axis=0)
        self.test_elemwise_uop(uops.argmin, np.argmin, axis=1)

    def test_argmax(self):
        self.test_elemwise_uop(uops.argmax, np.argmax)
        self.test_elemwise_uop(uops.argmax, np.argmax, axis=0)
        self.test_elemwise_uop(uops.argmax, np.argmax, axis=1)

    def test_min(self):
        self.test_elemwise_uop(uops.min, np.min)
        self.test_elemwise_uop(uops.min, np.min, axis=0)
        self.test_elemwise_uop(uops.min, np.min, axis=1)

    def test_max(self):
        self.test_elemwise_uop(uops.max, np.max)
        self.test_elemwise_uop(uops.max, np.max, axis=0)
        self.test_elemwise_uop(uops.max, np.max, axis=1)

    def test_norm(self):
        self.test_elemwise_uop(uops.norm, np.norm)
        self.test_elemwise_uop(uops.norm, np.norm, axis=0)
        self.test_elemwise_uop(uops.norm, np.norm, axis=1)

    def test_abs(self):
        self.test_elemwise_uop(uops.abs, np.abs)

    def test_neg(self):
        self.test_elemwise_uop(uops.neg, np.neg)

    def test_square(self):
        self.test_elemwise_uop(uops.square, np.square)

    def test_sqrt(self):
        self.test_elemwise_uop(uops.sqrt, np.sqrt)

    def test_sin(self):
        self.test_elemwise_uop(uops.sin, np.sin)

    def test_cos(self):
        self.test_elemwise_uop(uops.cos, np.cos)

    def test_tan(self):
        self.test_elemwise_uop(uops.tan, np.tan)

    def test_exp(self):
        self.test_elemwise_uop(uops.exp, np.exp)

    def test_sign(self):
        self.test_elemwise_uop(uops.sign, np.sign)






