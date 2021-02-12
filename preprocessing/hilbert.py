import numpy as np
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin


"""
Implements Sakhavi 2018
(https://ieeexplore.ieee.org/document/8310961)
section III, sub-section B, item 2.
"""
class Hilbert(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.abs(hilbert(X))
