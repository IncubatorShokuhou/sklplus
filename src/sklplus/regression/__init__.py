"""Regression task hub. Boost symbols need extra ``boost``."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklplus.dummy import DummyRegressor
from sklplus.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklplus.kernel_ridge import KernelRidge
from sklplus.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    TheilSenRegressor,
)
from sklplus.neighbors import KNeighborsRegressor
from sklplus.neural_network import MLPRegressor
from sklplus.svm import SVR
from sklplus.tree import DecisionTreeRegressor

_BOOST_EXPORTS: dict[str, tuple[str, str]] = {
    "CatBoostRegressor": ("sklplus.catboost", "CatBoostRegressor"),
    "LGBMRegressor": ("sklplus.lightgbm", "LGBMRegressor"),
    "XGBRegressor": ("sklplus.xgboost", "XGBRegressor"),
}

__all__ = [
    "ARDRegression",
    "AdaBoostRegressor",
    "BaggingRegressor",
    "BayesianRidge",
    "CatBoostRegressor",
    "DecisionTreeRegressor",
    "DummyRegressor",
    "ElasticNet",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HuberRegressor",
    "KNeighborsRegressor",
    "KernelRidge",
    "LGBMRegressor",
    "Lars",
    "Lasso",
    "LassoLars",
    "LinearRegression",
    "MLPRegressor",
    "OrthogonalMatchingPursuit",
    "PassiveAggressiveRegressor",
    "RANSACRegressor",
    "RandomForestRegressor",
    "Ridge",
    "SVR",
    "StackingRegressor",
    "TheilSenRegressor",
    "VotingRegressor",
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
