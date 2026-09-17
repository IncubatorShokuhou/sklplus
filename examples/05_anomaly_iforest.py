"""Anomaly Pipeline: StandardScaler + IForest."""

import numpy as np

from sklearnplus.anomaly import IForest
from sklearnplus.pipeline import Pipeline
from sklearnplus.preprocessing import StandardScaler


def main() -> int:
    rng = np.random.RandomState(0)
    X = rng.randn(120, 4)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("od", IForest(contamination=0.1, n_estimators=50, random_state=0)),
        ]
    )
    pipe.fit(X)
    pred = pipe.predict(X)
    n_flagged = int(np.sum(pred == 1))
    print(f"anomaly preds shape={pred.shape}; pyod_outlier_count={n_flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
