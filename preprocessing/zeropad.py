import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

"""
Pads the last `n_axes` axes of input with `pad_width` zeros on each side.
"""
class ZeroPad(TransformerMixin, BaseEstimator):
    def __init__(self, pad_width=4, n_axes=2):
        self.pad_width = pad_width
        self.n_axes = n_axes

    def fit(self, X, y):
        return self

    def transform(self, X):
        pad_widths = len(X.shape) * [(0, 0)]
        for i in range(0, self.n_axes):
            pad_widths[-(i+1)] = (self.pad_width, self.pad_width)

        padded = np.pad(X, pad_widths)

        return padded
