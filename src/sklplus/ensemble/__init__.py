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

from sklplus.catboost import CatBoostClassifier, CatBoostRegressor
from sklplus.lightgbm import LGBMClassifier, LGBMRegressor
from sklplus.xgboost import XGBClassifier, XGBRegressor

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
