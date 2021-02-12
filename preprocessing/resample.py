from scipy.signal import resample
from sklearn.base import BaseEstimator, TransformerMixin


"""
Implements Sakhavi 2018
(https://ieeexplore.ieee.org/document/8310961)
section III, sub-section B, item 4.
"""
class Resample(TransformerMixin, BaseEstimator):
    def __init__(self, old_fs=250, new_fs=10):
        self.old_fs = old_fs
        self.new_fs = new_fs

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_samples_new = round( X.shape[2] * self.new_fs / self.old_fs )
        return resample(X, n_samples_new, axis=2)
