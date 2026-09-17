"""Sanitize DataFrame column names for safe downstream use."""

from __future__ import annotations

import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CleanColumnNames(BaseEstimator, TransformerMixin):
    """Replace non-alphanumeric/underscore characters in column names with `_`."""

    _pattern = re.compile(r"[^0-9a-zA-Z_]")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CleanColumnNames expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = [self._pattern.sub("_", str(c)) for c in X.columns]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CleanColumnNames expects a pandas DataFrame")
        out = X.copy()
        out.columns = [
            self._pattern.sub("_", str(c)) for c in X.columns
        ]
        return out

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)
