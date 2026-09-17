"""End-to-end Pipeline smokes using sklplus imports only."""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, make_blobs


def test_clf_pipeline():
    from sklplus.ensemble import RandomForestClassifier
    from sklplus.pipeline import Pipeline
    from sklplus.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.9


def test_reg_pipeline():
    from sklplus.ensemble import RandomForestRegressor
    from sklplus.pipeline import Pipeline
    from sklplus.preprocessing import StandardScaler

    X, y = load_diabetes(return_X_y=True)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=10, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)
    assert pipe.score(X, y) > 0.5


def test_cluster_pipeline():
    from sklplus.cluster import KMeans
    from sklplus.pipeline import Pipeline
    from sklplus.preprocessing import StandardScaler

    X, _ = make_blobs(n_samples=120, centers=3, n_features=4, random_state=0)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=3, n_init=10, random_state=0)),
        ]
    )
    labels = pipe.fit_predict(X)
    assert labels.shape == (120,)
    assert len(set(labels.tolist())) == 3


def test_anomaly_pipeline():
    from sklplus.anomaly import IForest
    from sklplus.pipeline import Pipeline
    from sklplus.preprocessing import StandardScaler

    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("od", IForest(contamination=0.1, random_state=0)),
        ]
    )
    pipe.fit(X)
    pred = pipe.predict(X)
    assert pred.shape == (100,)
