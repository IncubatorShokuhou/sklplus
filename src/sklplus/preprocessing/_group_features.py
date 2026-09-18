"""Row-wise aggregate stats over named column groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklplus._tags import dataframe_only_tags


_AGG_FUNCS = {
    "min": np.nanmin,
    "max": np.nanmax,
    "mean": np.nanmean,
    "std": np.nanstd,
    "median": np.nanmedian,
    "mode": None,  # handled specially
}


def _row_mode(arr: np.ndarray) -> np.ndarray:
    """Mode along axis=1; ties break toward the first max-count value."""
    out = np.empty(arr.shape[0], dtype=float)
    for i, row in enumerate(arr):
        vals = row[~np.isnan(row)]
        if vals.size == 0:
            out[i] = np.nan
            continue
        uniq, counts = np.unique(vals, return_counts=True)
        out[i] = uniq[np.argmax(counts)]
    return out


class GroupFeatures(TransformerMixin, BaseEstimator):
    """Emit per-row aggregate statistics for named column groups.

    Parameters
    ----------
    groups :
        Mapping of group name -> sequence of column names. For each group and
        each aggregation, a column ``{group}_{agg}`` is added.
    aggregations :
        Aggregation names from ``{min, max, mean, std, median, mode}``.
    drop_original :
        If True, drop the source columns that appear in any group.
    """


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return dataframe_only_tags(tags)

    def __init__(
        self,
        groups: Mapping[str, Sequence[str]] | None = None,
        aggregations: Sequence[str] = ("min", "max", "mean", "std", "median"),
        drop_original: bool = False,
    ):
        self.groups = groups
        self.aggregations = aggregations
        self.drop_original = drop_original

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupFeatures expects a pandas DataFrame")
        if not self.groups:
            raise ValueError("GroupFeatures requires a non-empty groups mapping")
        unknown = [a for a in self.aggregations if a not in _AGG_FUNCS]
        if unknown:
            raise ValueError(f"Unsupported aggregations: {unknown}")
        self.feature_names_in_ = list(X.columns)
        self.groups_ = {str(k): list(v) for k, v in self.groups.items()}
        for name, cols in self.groups_.items():
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError(f"Group {name!r} missing columns: {missing}")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupFeatures expects a pandas DataFrame")
        out = X.copy()
        drop_cols: list[str] = []
        for gname, cols in self.groups_.items():
            block = out[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            for agg in self.aggregations:
                if agg == "mode":
                    values = _row_mode(block)
                else:
                    values = _AGG_FUNCS[agg](block, axis=1)
                out[f"{gname}_{agg}"] = values
            if self.drop_original:
                drop_cols.extend(cols)
        if drop_cols:
            out = out.drop(columns=list(dict.fromkeys(drop_cols)))
        return out
