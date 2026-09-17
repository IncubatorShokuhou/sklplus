"""Boosting dual-path: ensemble and xgboost export the same class."""

import numpy as np

from sklearnplus.ensemble import XGBClassifier as EnsembleXGB
from sklearnplus.xgboost import XGBClassifier as ModuleXGB


def main() -> int:
    assert EnsembleXGB is ModuleXGB, "dual-path must share the same class object"
    rng = np.random.RandomState(0)
    X = rng.randn(80, 6)
    y = (X[:, 0] > 0).astype(int)
    clf = EnsembleXGB(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        verbosity=0,
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    print(f"dual-path OK; tiny fit preds shape={pred.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
