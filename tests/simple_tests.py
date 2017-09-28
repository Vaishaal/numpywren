import sklearn.datasets as datasets
from numpywren.matrix import Matrix
from numpywren import matrix_utils
import pytest
import numpy as np

def test_single_shard_matrix():
    data = datasets.load_digits()
    X = data.data
    X_sharded = Matrix("test", shape=X.shape, shard_sizes=X.shape, bucket="numpywrentest")
    X_sharded.shard_matrix(X)
    assert(np.all(X_sharded.get_block(0,0) == X))

def test_multiple_shard_matrix():
    data = datasets.load_digits()
    X = data.data
    X_sharded = Matrix("test", shape=X.shape, shard_sizes=[128,32], bucket="numpywrentest")
    X_sharded.shard_matrix(X)
    X_sharded_local = matrix_utils.get_local_matrix(X_sharded)
    assert(np.all(X == X_sharded_local))


