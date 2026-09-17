import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from sklplus.preprocessing import (
    CleanColumnNames,
    DateFeatureExtractor,
    RareCategoryGrouper,
    StandardScaler,
    TargetLabelEncoder,
)


def test_standard_scaler_is_sklearn():
    assert StandardScaler is SkStandardScaler


def test_clean_column_names_dataframe_roundtrip():
    df = pd.DataFrame({"A-b": [1, 2], "c d": [3, 4], "ok_col": [5, 6]})
    tr = CleanColumnNames()
    out = tr.fit_transform(df)
    assert list(out.columns) == ["A_b", "c_d", "ok_col"]
    assert out["A_b"].tolist() == [1, 2]
    assert isinstance(out, pd.DataFrame)


def test_date_feature_extractor():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-15", "2021-06-20", "2022-12-01"]),
            "x": [1.0, 2.0, 3.0],
        }
    )
    tr = DateFeatureExtractor(
        features=("year", "month", "day", "dayofweek"), drop_original=True
    )
    out = tr.fit_transform(df)
    assert "ts" not in out.columns
    assert "ts_year" in out.columns
    assert "ts_month" in out.columns
    assert "ts_day" in out.columns
    assert "ts_dayofweek" in out.columns
    assert out["ts_year"].tolist() == [2020, 2021, 2022]
    assert out["x"].tolist() == [1.0, 2.0, 3.0]
    assert isinstance(out, pd.DataFrame)


def test_rare_category_grouper():
    # "a" appears often; "b","c" are rare (<5%)
    values = ["a"] * 95 + ["b"] * 3 + ["c"] * 2
    df = pd.DataFrame({"cat": values, "num": range(100)})
    tr = RareCategoryGrouper(min_frequency=0.05, replacement="rare")
    out = tr.fit_transform(df)
    assert set(out["cat"].unique()) <= {"a", "rare"}
    assert (out["cat"] == "a").sum() == 95
    assert (out["cat"] == "rare").sum() == 5
    assert out["num"].tolist() == list(range(100))


def test_target_label_encoder_roundtrip():
    y = np.array(["dog", "cat", "dog", "bird"])
    enc = TargetLabelEncoder()
    enc.fit(y)
    yt = enc.transform(y)
    assert yt.dtype.kind in "iu"
    assert set(yt.tolist()) == {0, 1, 2}
    y_back = enc.inverse_transform(yt)
    assert list(y_back) == list(y)
