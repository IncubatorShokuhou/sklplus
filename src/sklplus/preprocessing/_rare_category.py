"""Group infrequent categories into a single replacement label."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklplus._tags import dataframe_only_tags


def _is_categorical_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or str(series.dtype) in {"category", "string", "str"}
    )


class RareCategoryGrouper(TransformerMixin, BaseEstimator):
    """Replace rare levels in object/category/string columns.

    Parameters
    ----------
    min_frequency :
        Minimum relative frequency to keep a category as-is.
    replacement :
        Label used for grouped rare categories.
    """


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return dataframe_only_tags(tags)

    def __init__(self, min_frequency: float = 0.05, replacement: str = "rare"):
        self.min_frequency = min_frequency
        self.replacement = replacement

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RareCategoryGrouper expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self.categorical_columns_ = [c for c in X.columns if _is_categorical_like(X[c])]
        self.mapping_ = {}
        n = len(X)
        for col in self.categorical_columns_:
            freqs = X[col].astype("object").value_counts(dropna=False) / max(n, 1)
            keep = set(freqs[freqs >= self.min_frequency].index.tolist())
            self.mapping_[col] = keep
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RareCategoryGrouper expects a pandas DataFrame")
        out = X.copy()
        for col, keep in self.mapping_.items():
            series = out[col].astype("object")
            out[col] = series.where(series.isin(keep), other=self.replacement)
        return out
