"""Vectorize text columns and concatenate features back onto the frame."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextEmbedder(BaseEstimator, TransformerMixin):
    """Apply BoW or TF-IDF to text columns and concatenate dense features.

    Parameters
    ----------
    columns :
        Text columns to vectorize. If None, all object/string columns are used.
    method :
        ``"bow"`` (CountVectorizer) or ``"tfidf"`` (TfidfVectorizer).
    max_features :
        Passed to the underlying vectorizer.
    drop_original :
        If True, drop the source text columns after embedding.
    dense :
        If True (default), convert sparse vectorizer output to a dense
        DataFrame for easy concatenation. Set False to keep a sparse matrix
        for the text block only (still returns a DataFrame of dense values
        when other columns exist — sparse path returns dense for simplicity
        in v1).
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        method: str = "tfidf",
        max_features: int | None = 100,
        drop_original: bool = True,
        dense: bool = True,
    ):
        self.columns = None if columns is None else list(columns)
        self.method = method
        self.max_features = max_features
        self.drop_original = drop_original
        self.dense = dense

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TextEmbedder expects a pandas DataFrame")
        if self.method not in {"bow", "tfidf"}:
            raise ValueError("method must be 'bow' or 'tfidf'")
        self.feature_names_in_ = list(X.columns)
        if self.columns is None:
            self.text_columns_ = [
                c
                for c in X.columns
                if pd.api.types.is_object_dtype(X[c])
                or pd.api.types.is_string_dtype(X[c])
            ]
        else:
            missing = [c for c in self.columns if c not in X.columns]
            if missing:
                raise ValueError(f"Text columns missing: {missing}")
            self.text_columns_ = list(self.columns)

        Vectorizer = CountVectorizer if self.method == "bow" else TfidfVectorizer
        self.vectorizers_ = {}
        for col in self.text_columns_:
            vec = Vectorizer(max_features=self.max_features)
            texts = X[col].fillna("").astype(str)
            vec.fit(texts)
            self.vectorizers_[col] = vec
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TextEmbedder expects a pandas DataFrame")
        pieces: list[pd.DataFrame] = []
        other_cols = [c for c in X.columns if c not in self.text_columns_]
        if other_cols:
            pieces.append(X[other_cols].copy())
        elif not self.drop_original:
            pieces.append(X[self.text_columns_].copy())

        for col, vec in self.vectorizers_.items():
            texts = X[col].fillna("").astype(str)
            mat = vec.transform(texts)
            if self.dense or not sparse.issparse(mat):
                arr = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
            else:
                arr = mat.toarray()
            names = [f"{col}_{n}" for n in vec.get_feature_names_out()]
            pieces.append(pd.DataFrame(arr, columns=names, index=X.index))

        if not self.drop_original and self.text_columns_:
            # already included only when there were no other cols; otherwise add
            if other_cols:
                pieces.insert(1, X[self.text_columns_].copy())

        if not pieces:
            return pd.DataFrame(index=X.index)
        return pd.concat(pieces, axis=1)
