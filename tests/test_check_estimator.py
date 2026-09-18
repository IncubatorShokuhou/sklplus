"""check_estimator / clone coverage for custom estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from sklplus.preprocessing import (
    CleanColumnNames,
    DateFeatureExtractor,
    GroupFeatures,
    IterativeImputerPlus,
    RareCategoryGrouper,
    RemoveMulticollinearity,
    TargetLabelEncoder,
    TextEmbedder,
)
from sklplus.sampling import RemoveOutliers

# DataFrame-only transformers: tagged with two_d_array=False so ndarray
# check_estimator suites are skipped; remaining checks (cloneability) must pass.
_DF_TRANSFORMERS = [
    CleanColumnNames(),
    RareCategoryGrouper(),
    DateFeatureExtractor(),
    RemoveMulticollinearity(),
    IterativeImputerPlus(max_iter=2, random_state=0),
    TextEmbedder(columns=["text"], max_features=10),
    GroupFeatures(groups={"g": ["a", "b"]}, aggregations=("mean", "max")),
]


@pytest.mark.parametrize(
    "estimator",
    _DF_TRANSFORMERS,
    ids=lambda est: type(est).__name__,
)
def test_dataframe_transformer_check_estimator(estimator):
    """DF-only: ndarray checks skipped via tags; cloneability must pass."""
    check_estimator(estimator)


@pytest.mark.parametrize(
    "estimator",
    _DF_TRANSFORMERS,
    ids=lambda est: type(est).__name__,
)
def test_dataframe_transformer_clone(estimator):
    cloned = clone(estimator)
    assert type(cloned) is type(estimator)
    assert cloned.get_params() == estimator.get_params()


def test_target_label_encoder_clone_and_check_estimator():
    """y-only encoder: two_d_array=False skips X checks; clone must work."""
    enc = TargetLabelEncoder()
    check_estimator(enc)
    cloned = clone(enc)
    y = np.array(["a", "b", "a"])
    enc.fit(y)
    cloned2 = clone(enc)
    assert not hasattr(cloned2, "encoder_")
    assert list(cloned.fit(y).transform(y)) == list(enc.transform(y))


def test_remove_outliers_clone_and_check_estimator():
    """Sampler: check_estimator runs limited checks; clone + fit_resample smoke."""
    est = RemoveOutliers(method="iforest", contamination=0.1, random_state=0)
    check_estimator(est)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    X = np.random.RandomState(0).randn(40, 3)
    y = np.zeros(40)
    X2, y2 = cloned.fit_resample(X, y)
    assert len(X2) == len(y2) <= len(X)


def test_anomaly_adapters_clone_smoke():
    from sklplus.anomaly import HBOS, IForest, KNN

    for Cls in (IForest, HBOS, KNN):
        est = Cls()
        cloned = clone(est)
        assert type(cloned) is Cls
        assert cloned.get_params() == est.get_params()


@parametrize_with_checks(
    [
        CleanColumnNames(),
        RareCategoryGrouper(min_frequency=0.1),
        DateFeatureExtractor(features=("year", "month")),
        RemoveMulticollinearity(threshold=0.95),
        IterativeImputerPlus(max_iter=2, random_state=0),
        TextEmbedder(columns=["text"], max_features=5),
        GroupFeatures(groups={"g": ["a", "b"]}, aggregations=("mean",)),
    ]
)
def test_parametrize_with_checks_df_transformers(estimator, check):
    check(estimator)


def test_dataframe_transformers_accept_dataframe_roundtrip():
    """Behavioral smoke beyond check_estimator (which skips DF inputs)."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "cat": ["x", "x", "y", "z"],
            "text": ["hello world", "foo bar", "hello", "baz"],
            "ts": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ),
        }
    )
    assert isinstance(CleanColumnNames().fit(df).transform(df), pd.DataFrame)
    assert isinstance(
        RareCategoryGrouper(min_frequency=0.3).fit(df).transform(df), pd.DataFrame
    )
    assert isinstance(DateFeatureExtractor().fit(df).transform(df), pd.DataFrame)
    assert isinstance(
        GroupFeatures(groups={"g": ["a", "b"]}, aggregations=("mean",))
        .fit(df)
        .transform(df),
        pd.DataFrame,
    )
    assert isinstance(
        RemoveMulticollinearity(threshold=0.99).fit(df).transform(df), pd.DataFrame
    )
    assert isinstance(
        IterativeImputerPlus(max_iter=2, random_state=0).fit(df).transform(df),
        pd.DataFrame,
    )
    assert isinstance(
        TextEmbedder(columns=["text"], max_features=5).fit(df).transform(df),
        pd.DataFrame,
    )
