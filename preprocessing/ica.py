import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA

"""
Implements Jeong (2020)
section II, sub-section B, paragraph 1.

Jeong also removed some channels during the
ICA process but it doesn't apply on
our dataset (BNCI Horizon 2020 001-2014) bc
the excluded channels don't exist there. Jeong
used dataset 001-2017.
"""
class ICA(TransformerMixin, BaseEstimator):
    def __init__(self, n_out=None):
        self.n_out = n_out

    def fit(self, X, y):
        # Common
        n_epochs, n_channels, n_signalsamples = X.shape

        # Flatten epochs
        X = np.swapaxes(X, 1, 2)
        X = np.reshape(X, (n_epochs * n_signalsamples, n_channels))

        # Fit ICA
        ica = FastICA(n_components=self.n_out)
        ica.fit(X, None)

        # Save state
        self.ica = ica

        return self


    def transform(self, X):
        # Common
        n_epochs, n_channels, n_signalsamples = X.shape
        n_out = self.n_out or n_channels

        # Flatten epochs
        X = np.swapaxes(X, 1, 2)
        X = np.reshape(X, (n_epochs * n_signalsamples, n_channels))

        # Transform
        y = self.ica.transform(X)

        # Unflatten
        y = np.reshape(y, (n_epochs, n_signalsamples, n_out))
        y = np.swapaxes(y, 2, 1)

        return y
