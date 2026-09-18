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


    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # y-only target encoder: not an X→X transformer for check_estimator.
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        tags.target_tags.one_d_labels = True
        return tags

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
