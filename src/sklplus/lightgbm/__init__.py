"""lightgbm re-exports (optional extra: boost)."""

from __future__ import annotations

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    raise missing_extra_error("lightgbm", "boost") from exc

__all__ = ["LGBMClassifier", "LGBMRegressor"]
