import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklplus.anomaly import HBOS, IForest, KNN


def test_iforest_in_pipeline():
    X = np.random.randn(80, 4)
    pipe = Pipeline([("scaler", StandardScaler()), ("od", IForest(contamination=0.1))])
    pipe.fit(X)
    pred = pipe.predict(X)
    assert pred.shape == (80,)


@pytest.mark.parametrize("Detector", [IForest, HBOS, KNN])
def test_detector_pipeline_and_clone(Detector):
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    est = Detector(contamination=0.1)
    cloned = clone(est)
    pipe = Pipeline([("scaler", StandardScaler()), ("od", cloned)])
    pipe.fit(X)
    pred = pipe.predict(X)
    scores = pipe.decision_function(X)
    assert pred.shape == (80,)
    assert scores.shape == (80,)


def test_cblof_in_pipeline():
    from sklplus.anomaly import CBLOF

    rng = np.random.RandomState(1)
    X = rng.randn(80, 4)
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("od", CBLOF(contamination=0.1, n_clusters=8))]
    )
    pipe.fit(X)
    assert pipe.predict(X).shape == (80,)
