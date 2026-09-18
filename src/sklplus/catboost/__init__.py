"""catboost re-exports (optional extra: boost)."""

from __future__ import annotations

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    raise missing_extra_error("catboost", "boost") from exc

__all__ = ["CatBoostClassifier", "CatBoostRegressor"]
