import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Credit to: https://stackoverflow.com/questions/40394775/vectorizing-numpy-covariance-for-3d-array
def _batch_cov(x):
    N = x.shape[2]
    m1 = x - x.sum(2,keepdims=1)/N
    y_out = np.einsum('ijk,ilk->ijl',m1,m1) /(N - 1)
    return y_out


"""
Calculates covariance based on the last two elements.
Can be used on 3D {epoch, [csp_]channel, sample} or
4D (i.e. filter bank) {epoch, filter, [csp_]channel, sample} data.
"""
class Covariance(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        original_shape = X.shape

        # Flatten 4D into 3D if necessary
        if len(original_shape) == 4:
            X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
        
        # Calculate covariance matrices
        C = _batch_cov(X)

        # Un-flatten if necessary
        if len(original_shape) == 4:
            C = np.reshape(C, (original_shape[0], original_shape[1],
                original_shape[2], original_shape[2])) # No, it's not a typo.

        return C
