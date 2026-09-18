"""Iterative imputation for numeric columns plus simple impute for categoricals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklplus._tags import dataframe_only_tags
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer


def _is_categorical_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or str(series.dtype) in {"category", "string", "str"}
    )


class IterativeImputerPlus(TransformerMixin, BaseEstimator):
    """Impute numeric columns with ``IterativeImputer``; categoricals with mode.

    Limitation
    ----------
    Categorical columns are filled with the most frequent value only — they are
    not part of the iterative numeric model. Ordinal-encode-then-impute is not
    done in this v1.
    """


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return dataframe_only_tags(tags)

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int | None = None,
        categorical_strategy: str = "most_frequent",
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.categorical_strategy = categorical_strategy

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IterativeImputerPlus expects a pandas DataFrame")
        self.feature_names_in_ = list(X.columns)
        self.numeric_columns_ = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_columns_ = [c for c in X.columns if _is_categorical_like(X[c])]

        self.numeric_imputer_ = None
        if self.numeric_columns_:
            self.numeric_imputer_ = IterativeImputer(
                max_iter=self.max_iter, random_state=self.random_state
            )
            self.numeric_imputer_.fit(X[self.numeric_columns_])

        self.categorical_imputer_ = None
        if self.categorical_columns_:
            self.categorical_imputer_ = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer_.fit(X[self.categorical_columns_].astype("object"))
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("IterativeImputerPlus expects a pandas DataFrame")
        out = X.copy()
        if self.numeric_imputer_ is not None and self.numeric_columns_:
            imputed = self.numeric_imputer_.transform(out[self.numeric_columns_])
            out[self.numeric_columns_] = imputed
        if self.categorical_imputer_ is not None and self.categorical_columns_:
            imputed_cat = self.categorical_imputer_.transform(
                out[self.categorical_columns_].astype("object")
            )
            out[self.categorical_columns_] = imputed_cat
        return out
