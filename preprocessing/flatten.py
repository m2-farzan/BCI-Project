import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Flatten(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        if len(X.shape) == 3:
            return np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        elif len(X.shape) == 4:
            return np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        else:
            raise "Max dim = 4"
