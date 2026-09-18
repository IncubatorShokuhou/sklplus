"""Drop highly correlated numeric features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklplus._tags import dataframe_only_tags


class RemoveMulticollinearity(TransformerMixin, BaseEstimator):
    """Drop one of each highly correlated numeric feature pair.

    Parameters
    ----------
    threshold :
        Absolute Pearson correlation above which a pair is collinear.
    prefer_target :
        If True and ``y`` is provided at ``fit``, among a correlated pair keep
        the feature with higher absolute correlation to ``y``.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return dataframe_only_tags(tags)

    def __init__(self, threshold: float = 0.9, prefer_target: bool = True):
        self.threshold = threshold
        self.prefer_target = prefer_target

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RemoveMulticollinearity expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        numeric = X.select_dtypes(include=[np.number])
        self.numeric_columns_ = list(numeric.columns)
        if len(self.numeric_columns_) < 2:
            self.to_drop_ = []
            self.feature_names_out_ = list(X.columns)
            return self

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        target_corr = None
        if self.prefer_target and y is not None:
            y_ser = pd.Series(np.asarray(y).ravel(), index=X.index)
            if pd.api.types.is_numeric_dtype(y_ser):
                target_corr = numeric.corrwith(y_ser).abs()

        to_drop: set[str] = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            partners = upper.index[upper[col] > self.threshold].tolist()
            for other in partners:
                if other in to_drop:
                    continue
                if target_corr is not None:
                    # drop the one less correlated with y
                    drop = other if target_corr.get(col, 0) >= target_corr.get(other, 0) else col
                else:
                    # stable default: drop the later column
                    drop = other
                to_drop.add(drop)
                if drop == col:
                    break

        self.to_drop_ = [c for c in self.numeric_columns_ if c in to_drop]
        self.feature_names_out_ = [c for c in X.columns if c not in to_drop]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RemoveMulticollinearity expects a pandas DataFrame")
        return X.drop(columns=self.to_drop_, errors="ignore")
