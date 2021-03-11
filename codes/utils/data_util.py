import numpy as np


def train_test_split(X, y, test_size=0.2, seed=None):
    X, y = np.array(X), np.array(y)

    indices = np.arange(len(y))
    np.random.seed(seed)
    np.random.shuffle(indices)
    p = int(len(y) * (1 - test_size))
    train_ids, test_ids = indices[:p], indices[p:]

    return (X[train_ids], y[train_ids]), (X[test_ids], y[test_ids])
