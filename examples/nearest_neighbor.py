import numpywren as npywren
from npywren import npywrencc
import numpy as np
import sklearn.metrics as metrics

@npywrencc
def nearest_neighbor_numpy(X_train, X_test, y_train, y_test):
    XYT = np.dot(X_train.dot, X_test.T)
    norms_train = np.linalg.norm(X_train, axis=1)[:, np.newaxis]
    norms_test = np.linalg.norm(X_test, axis=1)[:, np.newaxis]
    XYT *= -2
    distances = norms_train + XYT + norms_test.T
    argmins = np.argmin(distances, axis=0)
    return metrics.accuracy_score(y_train[argmins], y_test)


def nearest_neighbor_numpywren(X_train, X_test, y_train, y_test):
    npwex = npywren.default_executor()
    X_train_sharded = npwex.matrix_init(X_train)
    X_test_sharded = npwex.matrix_init(X_test)
    XYT = npwex.dot(X_train_sharded, X_test_sharded.T)
    XYT *= -2
    norms_train = npywren.linalg.norm(X_train, axis=1)
    norms_test = npywren.linalg.norm(X_test, axis=1)
    distances = norms_train + XYT + norms_test.T
    argmins = npwex.argmin(distances, axis=0).numpy()
    return metrics.accuracy_score(y_train[argmins], y_test)

