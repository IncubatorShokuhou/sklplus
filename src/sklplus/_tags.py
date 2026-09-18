"""sklearn tags helpers for sklplus estimators."""

from __future__ import annotations


def dataframe_only_tags(tags):
    """Mark an estimator as pandas-DataFrame-only for check_estimator.

    Sets ``input_tags.two_d_array=False`` so the common ndarray-based
    estimator checks are skipped (honest for our DF-only transformers).
    """
    tags.input_tags.two_d_array = False
    tags.input_tags.allow_nan = True
    return tags
