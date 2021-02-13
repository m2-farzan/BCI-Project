# debug
import numpy as np
from sklearn.base import BaseEstimator


class Random(BaseEstimator):
    def __init__(self, n_classes=4):
        self.n_classes = n_classes
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(self.n_classes, size=X.shape[0])