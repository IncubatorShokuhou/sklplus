"""Classification Pipeline: StandardScaler + RandomForestClassifier."""

import numpy as np

from sklplus.ensemble import RandomForestClassifier
from sklplus.pipeline import Pipeline
from sklplus.preprocessing import StandardScaler


def main() -> int:
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=20, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    score = pipe.score(X, y)
    print(f"classification accuracy: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
