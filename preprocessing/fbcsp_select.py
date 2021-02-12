from mne.decoding import CSP
import numpy as np
import pandas as pd
from scipy.signal import iirfilter, sosfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FBCSP_Select(TransformerMixin, BaseEstimator):
    def __init__(self,
                 Nw=2,
                 Ns=4,
                 filters=[(4*i, 4*(i+1)) for i in range(1, 10)],
                 fs=250,
                 rs=30,
                 ):
        self.Nw = Nw
        self.Ns = Ns
        self.filters = filters
        self.fs = fs
        self.rs = rs


    def fit(self, X, y):
        """Select spatially filtered channels based on 
        Sakhavi 2018 (https://ieeexplore.ieee.org/document/8310961)
        section III, sub-section A.

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
        """

        # Common code
        n_filters = len(self.filters)
        n_epochs, n_channels, n_signalsamples = X.shape
        n_csp = 2 * self.Nw
        n_features = 2 * self.Ns
        classes = pd.unique(y)
        n_classes = len(classes)

        # Apply filter bank
        filtered_signals = np.zeros((n_filters, n_epochs, n_channels, n_signalsamples))
        for i in range(n_filters):
            filter_object = iirfilter(2, self.filters[i], ftype='cheby2', rs=self.rs,
                                      btype='bandpass', output='sos', fs=self.fs)
            filtered_signals[i] = sosfilt(filter_object, X, axis=2)
        
        # Find CSP bases
        csp_transformers = n_filters * [CSP(n_components=n_csp)]
        for i in range(n_filters):
            csp_transformers[i].fit(filtered_signals[i], y)

        # Calculate CSP powers
        csp_powers = np.zeros((n_epochs, n_filters, n_csp))
        for i in range(n_filters):
            csp_transformers[i].set_params(transform_into='average_power')
            csp_powers[:, i, :] = csp_transformers[i].transform(filtered_signals[i])
        csp_powers = np.reshape(csp_powers, (n_epochs, n_filters * n_csp)) # flatten

        # Feature selection
        feature_selector = SelectKBest(mutual_info_classif, k=n_features)
        selected_features_indices = []
        for i in range(n_classes):
            y_masked = np.where(y == classes[i], True, False) # one-vs-other approach, explained in III.A.6
            feature_selector.fit(csp_powers, y_masked)
            selected_features_indices += list( feature_selector.get_support(indices=True) )

        # As feature selection is done separately for each class,
        # a feature may appear multiple times in the list. Here we
        # replace the repeated features in the list with other
        # features which are selected using a global (non-class-specific)
        # feature selector.
        global_feature_selector = SelectKBest(mutual_info_classif, k=(n_classes * n_features))
        global_feature_selector.fit(csp_powers, y)
        selected_features_indices += list( global_feature_selector.get_support(indices=True) )
        selected_features_indices = list(pd.unique(selected_features_indices))[:n_classes * n_features]

        # Save pipeline
        self.csp_transformers = csp_transformers
        self.selected_features_indices = selected_features_indices
        
        return self


    def transform(self, X):
        # Common code
        n_filters = len(self.filters)
        n_epochs, n_channels, n_signalsamples = X.shape
        n_csp = 2 * self.Nw
        
        # Load pipeline
        csp_transformers = self.csp_transformers
        selected_features_indices = self.selected_features_indices

        # Apply filter bank
        filtered_signals = np.zeros((n_filters, n_epochs, n_channels, n_signalsamples))
        for i in range(n_filters):
            filter_object = iirfilter(2, self.filters[i], ftype='cheby2', rs=self.rs,
                                      btype='bandpass', output='sos', fs=self.fs)
            filtered_signals[i] = sosfilt(filter_object, X, axis=2)

        # Calculate CSP signals
        csp_signals = np.zeros((n_epochs, n_filters, n_csp, n_signalsamples))
        for i in range(n_filters):
            csp_transformers[i].set_params(transform_into='csp_space')
            csp_signals[:, i, :, :] = csp_transformers[i].transform(filtered_signals[i])
        csp_signals = np.reshape(csp_signals, (n_epochs, n_filters * n_csp, n_signalsamples)) # flatten

        # Feature selection
        selected_csp_signals = csp_signals[:, selected_features_indices, :]

        return selected_csp_signals

