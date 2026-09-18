"""xgboost re-exports (optional extra: boost)."""

from __future__ import annotations

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    raise missing_extra_error("xgboost", "boost") from exc

__all__ = ["XGBClassifier", "XGBRegressor"]
