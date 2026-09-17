import numpy as np
import pandas as pd
import pytest

from sklplus.linear_model import LogisticRegression
from sklplus.pipeline import ImbPipeline
from sklplus.preprocessing import (
    GroupFeatures,
    IterativeImputerPlus,
    RemoveMulticollinearity,
    TextEmbedder,
)
from sklplus.sampling import RemoveOutliers


def test_group_features_dataframe_roundtrip():
    df = pd.DataFrame(
        {
            "a1": [1.0, 2.0, 3.0],
            "a2": [3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0],
        }
    )
    tr = GroupFeatures(
        groups={"g": ["a1", "a2"]},
        aggregations=("min", "max", "mean"),
        drop_original=False,
    )
    out = tr.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert "g_min" in out.columns and "g_max" in out.columns and "g_mean" in out.columns
    assert out["g_min"].tolist() == [1.0, 2.0, 3.0]
    assert out["g_max"].tolist() == [3.0, 4.0, 5.0]
    assert out["g_mean"].tolist() == [2.0, 3.0, 4.0]
    assert "a1" in out.columns and "b" in out.columns


def test_remove_multicollinearity_drops_correlated():
    rng = np.random.RandomState(0)
    x1 = rng.randn(80)
    x2 = x1 + 1e-6 * rng.randn(80)  # nearly identical
    x3 = rng.randn(80)
    y = (x1 + x3 > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat": ["a"] * 80})
    tr = RemoveMulticollinearity(threshold=0.95, prefer_target=True)
    out = tr.fit_transform(df, y)
    assert isinstance(out, pd.DataFrame)
    assert "cat" in out.columns
    # one of x1/x2 should be dropped
    assert len(set(out.columns) & {"x1", "x2"}) == 1
    assert "x3" in out.columns


def test_iterative_imputer_plus_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "n1": [1.0, np.nan, 3.0, 4.0],
            "n2": [2.0, 2.5, np.nan, 4.0],
            "c1": ["a", None, "a", "b"],
        }
    )
    tr = IterativeImputerPlus(max_iter=5, random_state=0)
    out = tr.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert not out["n1"].isna().any()
    assert not out["n2"].isna().any()
    assert not out["c1"].isna().any()
    assert set(out["c1"].unique()) <= {"a", "b"}


def test_text_embedder_tfidf_concat():
    df = pd.DataFrame(
        {
            "text": ["good movie", "bad movie", "good film", "bad film"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    tr = TextEmbedder(columns=["text"], method="tfidf", max_features=10, drop_original=True)
    out = tr.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert "text" not in out.columns
    assert "num" in out.columns
    assert out.shape[0] == 4
    assert out.shape[1] > 1  # num + embedding cols


def test_text_embedder_bow():
    df = pd.DataFrame({"doc": ["alpha beta", "beta gamma", "alpha"]})
    tr = TextEmbedder(columns=["doc"], method="bow", max_features=5, drop_original=True)
    out = tr.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 3
    assert all(c.startswith("doc_") for c in out.columns)


@pytest.mark.parametrize("method", ["iforest", "lof", "ee"])
def test_remove_outliers_fit_resample(method):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
    # inject clear outliers
    X.iloc[0] = [50.0, 50.0, 50.0]
    X.iloc[1] = [-50.0, -50.0, -50.0]
    y = np.zeros(100, dtype=int)
    y[50:] = 1
    sampler = RemoveOutliers(method=method, contamination=0.05, random_state=0)
    Xr, yr = sampler.fit_resample(X, y)
    assert isinstance(Xr, pd.DataFrame)
    assert len(Xr) == len(yr)
    assert len(Xr) < len(X)
    assert list(Xr.columns) == ["a", "b", "c"]


def test_remove_outliers_imbpipeline():
    rng = np.random.RandomState(1)
    X = rng.randn(120, 4)
    X[0] = 40.0
    y = (X[:, 0] > 0).astype(int)
    pipe = ImbPipeline(
        [
            ("out", RemoveOutliers(method="iforest", contamination=0.05, random_state=0)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)
