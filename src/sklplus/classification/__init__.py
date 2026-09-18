"""Classification task hub. Boost symbols need extra ``boost``."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklplus.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklplus.dummy import DummyClassifier
from sklplus.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    CalibratedClassifierCV,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklplus.gaussian_process import GaussianProcessClassifier
from sklplus.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklplus.naive_bayes import GaussianNB
from sklplus.neighbors import KNeighborsClassifier
from sklplus.neural_network import MLPClassifier
from sklplus.svm import SVC
from sklplus.tree import DecisionTreeClassifier

_BOOST_EXPORTS: dict[str, tuple[str, str]] = {
    "CatBoostClassifier": ("sklplus.catboost", "CatBoostClassifier"),
    "LGBMClassifier": ("sklplus.lightgbm", "LGBMClassifier"),
    "XGBClassifier": ("sklplus.xgboost", "XGBClassifier"),
}

__all__ = [
    "AdaBoostClassifier",
    "BaggingClassifier",
    "CalibratedClassifierCV",
    "CatBoostClassifier",
    "DecisionTreeClassifier",
    "DummyClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
    "LGBMClassifier",
    "LinearDiscriminantAnalysis",
    "LogisticRegression",
    "MLPClassifier",
    "QuadraticDiscriminantAnalysis",
    "RandomForestClassifier",
    "RidgeClassifier",
    "SGDClassifier",
    "SVC",
    "StackingClassifier",
    "VotingClassifier",
    "XGBClassifier",
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
