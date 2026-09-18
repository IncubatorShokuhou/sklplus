"""Ensemble estimators. Boosting libs need extra ``boost``."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

_BOOST_EXPORTS: dict[str, tuple[str, str]] = {
    "CatBoostClassifier": ("sklplus.catboost", "CatBoostClassifier"),
    "CatBoostRegressor": ("sklplus.catboost", "CatBoostRegressor"),
    "LGBMClassifier": ("sklplus.lightgbm", "LGBMClassifier"),
    "LGBMRegressor": ("sklplus.lightgbm", "LGBMRegressor"),
    "XGBClassifier": ("sklplus.xgboost", "XGBClassifier"),
    "XGBRegressor": ("sklplus.xgboost", "XGBRegressor"),
}

__all__ = [
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "CalibratedClassifierCV",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "XGBClassifier",
    "XGBRegressor",
]


def __getattr__(name: str) -> Any:
    if name in _BOOST_EXPORTS:
        module_name, attr = _BOOST_EXPORTS[name]
        mod = import_module(module_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
