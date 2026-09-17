"""ImbPipeline: SMOTE + LogisticRegression on a mildly imbalanced set."""

import numpy as np

from sklplus.linear_model import LogisticRegression
from sklplus.pipeline import ImbPipeline
from sklplus.sampling import SMOTE


def main() -> int:
    rng = np.random.RandomState(0)
    n_major, n_minor = 170, 30
    X_major = rng.randn(n_major, 10)
    X_minor = rng.randn(n_minor, 10) + 1.5
    X = np.vstack([X_major, X_minor])
    y = np.array([0] * n_major + [1] * n_minor)
    pipe = ImbPipeline(
        [
            ("sample", SMOTE(random_state=0)),
            ("clf", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    score = pipe.score(X, y)
    print(f"imbalanced accuracy after SMOTE: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
