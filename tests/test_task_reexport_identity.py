"""Multi-path identity: task hubs and ensemble technique submodules.

Re-export hubs must expose the same class objects as the existing sklearn-style
paths (`is`), without factories or moved implementations.
"""
import importlib

import pytest

CLASSIFICATION_SOURCES = [
    ("sklplus.linear_model", "LogisticRegression"),
    ("sklplus.linear_model", "RidgeClassifier"),
    ("sklplus.linear_model", "SGDClassifier"),
    ("sklplus.tree", "DecisionTreeClassifier"),
    ("sklplus.neighbors", "KNeighborsClassifier"),
    ("sklplus.svm", "SVC"),
    ("sklplus.naive_bayes", "GaussianNB"),
    ("sklplus.discriminant_analysis", "LinearDiscriminantAnalysis"),
    ("sklplus.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    ("sklplus.neural_network", "MLPClassifier"),
    ("sklplus.gaussian_process", "GaussianProcessClassifier"),
    ("sklplus.dummy", "DummyClassifier"),
    ("sklplus.ensemble", "RandomForestClassifier"),
    ("sklplus.ensemble", "ExtraTreesClassifier"),
    ("sklplus.ensemble", "AdaBoostClassifier"),
    ("sklplus.ensemble", "GradientBoostingClassifier"),
    ("sklplus.ensemble", "BaggingClassifier"),
    ("sklplus.ensemble", "StackingClassifier"),
    ("sklplus.ensemble", "VotingClassifier"),
    ("sklplus.ensemble", "CalibratedClassifierCV"),
    ("sklplus.ensemble", "XGBClassifier"),
    ("sklplus.ensemble", "LGBMClassifier"),
    ("sklplus.ensemble", "CatBoostClassifier"),
]

REGRESSION_SOURCES = [
    ("sklplus.linear_model", "LinearRegression"),
    ("sklplus.linear_model", "Ridge"),
    ("sklplus.linear_model", "Lasso"),
    ("sklplus.linear_model", "ElasticNet"),
    ("sklplus.linear_model", "Lars"),
    ("sklplus.linear_model", "LassoLars"),
    ("sklplus.linear_model", "OrthogonalMatchingPursuit"),
    ("sklplus.linear_model", "BayesianRidge"),
    ("sklplus.linear_model", "ARDRegression"),
    ("sklplus.linear_model", "PassiveAggressiveRegressor"),
    ("sklplus.linear_model", "RANSACRegressor"),
    ("sklplus.linear_model", "TheilSenRegressor"),
    ("sklplus.linear_model", "HuberRegressor"),
    ("sklplus.tree", "DecisionTreeRegressor"),
    ("sklplus.neighbors", "KNeighborsRegressor"),
    ("sklplus.svm", "SVR"),
    ("sklplus.neural_network", "MLPRegressor"),
    ("sklplus.dummy", "DummyRegressor"),
    ("sklplus.kernel_ridge", "KernelRidge"),
    ("sklplus.ensemble", "RandomForestRegressor"),
    ("sklplus.ensemble", "ExtraTreesRegressor"),
    ("sklplus.ensemble", "AdaBoostRegressor"),
    ("sklplus.ensemble", "GradientBoostingRegressor"),
    ("sklplus.ensemble", "BaggingRegressor"),
    ("sklplus.ensemble", "StackingRegressor"),
    ("sklplus.ensemble", "VotingRegressor"),
    ("sklplus.ensemble", "XGBRegressor"),
    ("sklplus.ensemble", "LGBMRegressor"),
    ("sklplus.ensemble", "CatBoostRegressor"),
]

BAGGING_SOURCES = [
    ("sklplus.ensemble", "BaggingClassifier"),
    ("sklplus.ensemble", "BaggingRegressor"),
]

BOOSTING_SOURCES = [
    ("sklplus.ensemble", "AdaBoostClassifier"),
    ("sklplus.ensemble", "AdaBoostRegressor"),
    ("sklplus.ensemble", "GradientBoostingClassifier"),
    ("sklplus.ensemble", "GradientBoostingRegressor"),
    ("sklplus.ensemble", "XGBClassifier"),
    ("sklplus.ensemble", "XGBRegressor"),
    ("sklplus.ensemble", "LGBMClassifier"),
    ("sklplus.ensemble", "LGBMRegressor"),
    ("sklplus.ensemble", "CatBoostClassifier"),
    ("sklplus.ensemble", "CatBoostRegressor"),
]

STACKING_SOURCES = [
    ("sklplus.ensemble", "StackingClassifier"),
    ("sklplus.ensemble", "StackingRegressor"),
]

VOTING_SOURCES = [
    ("sklplus.ensemble", "VotingClassifier"),
    ("sklplus.ensemble", "VotingRegressor"),
]


def _import_attr(module_path: str, name: str):
    return getattr(importlib.import_module(module_path), name)


def _ids(pairs):
    return [f"{mod}:{name}" for mod, name in pairs]


@pytest.mark.parametrize("source_mod,name", CLASSIFICATION_SOURCES, ids=_ids(CLASSIFICATION_SOURCES))
def test_classification_hub_identity(source_mod, name):
    hub = _import_attr("sklplus.classification", name)
    source = _import_attr(source_mod, name)
    assert hub is source


@pytest.mark.parametrize("source_mod,name", REGRESSION_SOURCES, ids=_ids(REGRESSION_SOURCES))
def test_regression_hub_identity(source_mod, name):
    hub = _import_attr("sklplus.regression", name)
    source = _import_attr(source_mod, name)
    assert hub is source


@pytest.mark.parametrize("source_mod,name", BAGGING_SOURCES, ids=_ids(BAGGING_SOURCES))
def test_ensemble_bagging_identity(source_mod, name):
    hub = _import_attr("sklplus.ensemble.bagging", name)
    source = _import_attr(source_mod, name)
    assert hub is source


@pytest.mark.parametrize("source_mod,name", BOOSTING_SOURCES, ids=_ids(BOOSTING_SOURCES))
def test_ensemble_boosting_identity(source_mod, name):
    hub = _import_attr("sklplus.ensemble.boosting", name)
    source = _import_attr(source_mod, name)
    assert hub is source


@pytest.mark.parametrize("source_mod,name", STACKING_SOURCES, ids=_ids(STACKING_SOURCES))
def test_ensemble_stacking_identity(source_mod, name):
    hub = _import_attr("sklplus.ensemble.stacking", name)
    source = _import_attr(source_mod, name)
    assert hub is source


@pytest.mark.parametrize("source_mod,name", VOTING_SOURCES, ids=_ids(VOTING_SOURCES))
def test_ensemble_voting_identity(source_mod, name):
    hub = _import_attr("sklplus.ensemble.voting", name)
    source = _import_attr(source_mod, name)
    assert hub is source


def test_xgb_classifier_three_path_identity():
    from sklplus.ensemble.boosting import XGBClassifier as from_boosting
    from sklplus.xgboost import XGBClassifier as from_xgboost
    from sklplus.ensemble import XGBClassifier as from_ensemble

    assert from_boosting is from_xgboost is from_ensemble


def test_xgb_classifier_also_via_classification_hub():
    from sklplus.classification import XGBClassifier as from_task
    from sklplus.xgboost import XGBClassifier as from_xgboost

    assert from_task is from_xgboost


def test_bagging_does_not_export_random_forest_or_extra_trees():
    bagging = importlib.import_module("sklplus.ensemble.bagging")
    leaked = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
    }
    exported = set(getattr(bagging, "__all__", dir(bagging)))
    assert leaked.isdisjoint(exported)


def test_no_top_level_stacking_package():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sklplus.stacking")
