"""Train-only outlier row filter as an imblearn-compatible sampler."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_consistent_length


class RemoveOutliers(BaseEstimator):
    """Drop outlier rows via ``fit_resample`` (imblearn / ``ImbPipeline``).

    Fits an unsupervised outlier detector on ``X`` and keeps inliers. Use with
    ``ImbPipeline`` so ``y`` is filtered in lockstep. Do **not** put this in a
    plain sklearn ``Pipeline`` as a transformer — there is no ``transform`` that
    filters test rows (by design, to avoid train/test leakage patterns).

    Parameters
    ----------
    method :
        ``"iforest"``, ``"lof"``, or ``"ee"`` (EllipticEnvelope).
    contamination :
        Expected fraction of outliers (passed to the underlying estimator).
    random_state :
        Random seed where supported (iforest / ee).
    """

    def __init__(
        self,
        method: str = "iforest",
        contamination: float = 0.05,
        random_state: int | None = None,
    ):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state

    def _make_detector(self):
        if self.method == "iforest":
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            )
        if self.method == "lof":
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=False,
            )
        if self.method == "ee":
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
            )
        raise ValueError("method must be one of {'iforest', 'lof', 'ee'}")

    def fit_resample(self, X, y):
        """Fit detector on ``X`` and return inlier ``(X, y)``."""
        was_df = isinstance(X, pd.DataFrame)
        columns = list(X.columns) if was_df else None
        index = X.index if was_df else None

        X_arr = check_array(X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True)
        y_arr = np.asarray(y)
        check_consistent_length(X_arr, y_arr)

        detector = self._make_detector()
        if self.method == "lof":
            labels = detector.fit_predict(X_arr)
        else:
            labels = detector.fit_predict(X_arr)
        self.detector_ = detector
        mask = labels == 1
        self.inlier_mask_ = mask
        self.n_features_in_ = X_arr.shape[1]

        X_keep = X_arr[mask]
        y_keep = y_arr[mask]
        if was_df:
            X_keep = pd.DataFrame(X_keep, columns=columns, index=np.asarray(index)[mask])
        return X_keep, y_keep

    def fit(self, X, y=None):
        """Fit only (stores detector); prefer ``fit_resample`` for filtering."""
        X_arr = check_array(X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True)
        detector = self._make_detector()
        if self.method == "lof":
            # LOF without novelty cannot predict later; still fit for API parity
            labels = detector.fit_predict(X_arr)
            self.inlier_mask_ = labels == 1
        else:
            detector.fit(X_arr)
        self.detector_ = detector
        self.n_features_in_ = X_arr.shape[1]
        return self
