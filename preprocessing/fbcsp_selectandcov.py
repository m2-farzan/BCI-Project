from mne.decoding import CSP
import numpy as np
import pandas as pd
from scipy.signal import iirfilter, sosfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif


FILTERS_YK2019 = [
    (7.5, 14), (11, 13), (10, 14), (9, 12), (19, 22), (16, 22), (26, 34),
    (17.5, 20.5), (7, 30), (5, 14), (11, 31), (12, 18), (7, 9), (15, 17),
    (25, 30), (20, 25), (5, 10), (10, 25), (15, 30), (10, 12), (23, 27),
    (28, 32), (12, 33), (11, 22), (5, 8), (7.5, 17.5), (23, 26), (5, 20),
    (5, 25), (10, 20)
] # from Yeon Kwon 2019 (https://ieeexplore.ieee.org/document/8897723) table I.


# Credit to: https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array
def _batch_cov(x):
    N = x.shape[2]
    m1 = x - x.sum(2,keepdims=1)/N
    y_out = np.einsum('ijk,ilk->ijl',m1,m1) /(N - 1)
    return y_out

# Todo: Refactor into CspSelect and Covariance, and wrap them into a FB Pipeline util.
class FBCSP_SelectAndCov(TransformerMixin, BaseEstimator):
    def __init__(self,
                 U=None,
                 P=20,
                 filters=FILTERS_YK2019,
                 fs=250,
                 rs=30,
                 ):
        self.U = U
        self.P = P
        self.filters = filters
        self.fs = fs
        self.rs = rs


    def fit(self, X, y):
        """Select spatially filtered channels based on 
        Yeon Kwon 2019 (https://ieeexplore.ieee.org/document/8897723)
        section IV, sub-section A.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The EEG signals.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of FBCSP_Select
            Returns the modified instance.

        Note:
        For cross-subject testing, each epoch (X[i, :, :]) can be concat of multiple epochs performed on a single subject
        """

        # Common code
        n_filters = len(self.filters)
        n_epochs, n_channels, n_signalsamples = X.shape
        n_csp = self.U or n_channels
        n_features = self.P

        # Common definitions
        csp_transformers = n_filters * [CSP(n_components=n_csp, cov_est='concat')]

        # Find band csp powers
        band_csp_powers = np.zeros((n_epochs, n_filters))

        for i in range(n_filters):
            # Apply filter bank
            filter_object = iirfilter(2, self.filters[i], ftype='butter', rs=self.rs,
                                      btype='bandpass', output='sos', fs=self.fs)
            filtered_signal = sosfilt(filter_object, X, axis=2)

            # Find CSP bases
            csp_transformers[i].fit(filtered_signal, y)

            # Calculate all csp powers
            csp_transformers[i].set_params(transform_into='average_power')
            all_csp_powers = csp_transformers[i].transform(filtered_signal)

            # Calculate total CSP power for each filter band (Paper #2, Section IV.A, Eq. 2)
            band_csp_powers[:, i] = np.sum(all_csp_powers, axis=1)

        # Feature selection
        feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        feature_selector.fit(band_csp_powers, y)
        selected_filter_indices = feature_selector.get_support(indices=True)

        # Save pipeline
        self.csp_transformers = csp_transformers
        self.selected_filter_indices = selected_filter_indices

        return self


    def transform(self, X):
        # Common code
        n_epochs, n_channels, n_signalsamples = X.shape
        n_csp = self.U or n_channels

        # Load pipeline
        csp_transformers = self.csp_transformers
        selected_filter_indices = self.selected_filter_indices
        n_selected_filters = len(selected_filter_indices)

        # Find Covariance Matrices
        covariances = np.zeros((n_epochs, n_selected_filters, n_csp, n_csp))
        for i in range(n_selected_filters):
            # Apply filter bank
            filter_object = iirfilter(2, self.filters[ selected_filter_indices[i] ], ftype='butter', rs=self.rs,
                                      btype='bandpass', output='sos', fs=self.fs)
            filtered_signal = sosfilt(filter_object, X, axis=2) # (n_epochs, n_channels, n_signalsamples)

            # Calculate CSP signals
            csp_transformers[ selected_filter_indices[i] ].set_params(transform_into='csp_space')
            selected_csp_signals = csp_transformers[ selected_filter_indices[i] ].transform(filtered_signal)

            # Calculate Covariance
            covariances[:, i, :, :] = _batch_cov(selected_csp_signals)

        return covariances


    def get_selected_band_filters(self):
        return list(
            np.asarray(self.filters)[self.selected_filter_indices]
        )
