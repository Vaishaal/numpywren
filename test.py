import numpy as np
import pywren
import time
import numpywren
from numpywren import matrix
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init
from numpywren.matrix_utils import chunk
X = np.random.randn(128,128)
X_sharded = local_numpy_init(X, X.shape)
local_numpy_init(X, shard_sizes=X.shape)
X_sharded_local = X_sharded.get_block(0,0)
X_sharded.free()
assert(np.all(X_sharded_local == X))
