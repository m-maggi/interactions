# From https://github.com/lorentzenchr/responsible_ml_material

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LogRegressor(RegressorMixin):
    """
    A wrapper class for a Scikit-Learn regressor that evaluates predictions on a log scale.

    Parameters
    ----------
    regressor : object
        A Scikit-Learn regressor object that has already been fit to data.
    
    Methods
    -------
    predict(X)
        Make predictions for the given input data X.

    fit(*args, **kwargs)
        Not used.
    """
    def __init__(self, estimator):
        self._estimator = estimator
        check_is_fitted(self._estimator)
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return np.log(self._estimator.predict(X))