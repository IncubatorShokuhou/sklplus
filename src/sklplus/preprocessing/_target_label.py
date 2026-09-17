"""Encode classification targets with a reversible label mapping."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


class TargetLabelEncoder(BaseEstimator):
    """Thin wrapper around ``LabelEncoder`` for classification targets.

    Primary API (as documented for y-only use)::

        enc.fit(y)
        yt = enc.transform(y)
        y_hat = enc.inverse_transform(yt)
    """

    def fit(self, y):
        self.encoder_ = LabelEncoder()
        self.encoder_.fit(y)
        self.classes_ = self.encoder_.classes_
        return self

    def transform(self, y):
        return self.encoder_.transform(y)

    def inverse_transform(self, y):
        return self.encoder_.inverse_transform(np.asarray(y))

    def fit_transform(self, y):
        return self.fit(y).transform(y)
