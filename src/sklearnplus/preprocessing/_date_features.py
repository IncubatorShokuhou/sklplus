"""Expand datetime columns into numeric calendar features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _is_datetime_like(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_datetime64_any_dtype(dtype)
        or isinstance(dtype, pd.DatetimeTZDtype)
    )


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract calendar parts from datetime columns.

    Parameters
    ----------
    features :
        Datetime attributes to extract (e.g. year, month, day, dayofweek).
    drop_original :
        If True, drop the source datetime columns after expansion.
    """

    def __init__(
        self,
        features: Sequence[str] = ("year", "month", "day", "dayofweek"),
        drop_original: bool = True,
    ):
        self.features = tuple(features)
        self.drop_original = drop_original

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateFeatureExtractor expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self.datetime_columns_ = [c for c in X.columns if _is_datetime_like(X[c])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateFeatureExtractor expects a pandas DataFrame")
        out = X.copy()
        for col in self.datetime_columns_:
            series = pd.to_datetime(out[col])
            for feat in self.features:
                out[f"{col}_{feat}"] = getattr(series.dt, feat)
            if self.drop_original:
                out = out.drop(columns=[col])
        return out
