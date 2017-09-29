import sklearn.datasets as datasets
from numpywren.matrix import BigMatrix
from numpywren import matrix_utils, ops
import pytest
import numpy as np
import pywren


def test_single_shard_matrix():
    np.random.seed(0)
    X = np.random.randn(128,128)
    X_sharded = BigMatrix("test_0", shape=X.shape, shard_sizes=X.shape)
    X_sharded.shard_matrix(X)
    X_sharded_local = X_sharded.numpy()
    X_sharded.free()
    assert(np.all(X_sharded_local == X))

def test_multiple_shard_matrix():
    np.random.seed(0)
    X = np.random.randn(128,128)
    shard_sizes = tuple(map(int, np.array(X.shape)/2))
    X_sharded = BigMatrix("test_1", shape=X.shape, shard_sizes=shard_sizes)
    X_sharded.shard_matrix(X)
    X_sharded_local = X_sharded.numpy()
    X_sharded.free()
    assert(np.all(X == X_sharded_local))

def test_single_shard_matrix_multiply():
    np.random.seed(0)
    X = np.random.randn(128,128)
    X_sharded = BigMatrix("test_2", shape=X.shape, shard_sizes=X.shape)
    X_sharded.shard_matrix(X)
    pwex = pywren.default_executor()
    XXT_sharded = ops.cxyt(pwex, X_sharded, X_sharded, X_sharded.bucket, 1)

    XXT_sharded_local = XXT_sharded.numpy()
    XXT = X.dot(X.T)
    X_sharded.free()
    XXT_sharded.free()
    assert(np.all(np.isclose(XXT,XXT_sharded_local)))

def test_multiple_shard_matrix_multiply():
    np.random.seed(0)
    X = np.random.randn(128,128)
    shard_sizes = tuple(map(int, np.array(X.shape)/2))
    X_sharded = BigMatrix("test_3", shape=X.shape, shard_sizes=shard_sizes)
    X_sharded.shard_matrix(X)
    pwex = pywren.default_executor()
    XXT_sharded = ops.cxyt(pwex, X_sharded, X_sharded, X_sharded.bucket, 1)
    XXT_sharded_local = XXT_sharded.numpy()
    XXT = X.dot(X.T)
    X_sharded.free()
    XXT_sharded.free()
    assert(np.all(np.isclose(XXT,XXT_sharded_local)))

if __name__ == "__main__":
    test_single_shard_matrix()
    test_multiple_shard_matrix()
    test_single_shard_matrix_multiply()
    test_multiple_shard_matrix_multiply()


