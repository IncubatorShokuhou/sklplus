"""Sampling utilities. imblearn re-exports need extra ``imblearn``."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sklplus.sampling._remove_outliers import RemoveOutliers

_IMBLEARN_EXPORTS: dict[str, tuple[str, str]] = {
    "ADASYN": ("imblearn.over_sampling", "ADASYN"),
    "AllKNN": ("imblearn.under_sampling", "AllKNN"),
    "BorderlineSMOTE": ("imblearn.over_sampling", "BorderlineSMOTE"),
    "CondensedNearestNeighbour": (
        "imblearn.under_sampling",
        "CondensedNearestNeighbour",
    ),
    "EditedNearestNeighbours": (
        "imblearn.under_sampling",
        "EditedNearestNeighbours",
    ),
    "InstanceHardnessThreshold": (
        "imblearn.under_sampling",
        "InstanceHardnessThreshold",
    ),
    "KMeansSMOTE": ("imblearn.over_sampling", "KMeansSMOTE"),
    "NearMiss": ("imblearn.under_sampling", "NearMiss"),
    "NeighbourhoodCleaningRule": (
        "imblearn.under_sampling",
        "NeighbourhoodCleaningRule",
    ),
    "OneSidedSelection": ("imblearn.under_sampling", "OneSidedSelection"),
    "RandomOverSampler": ("imblearn.over_sampling", "RandomOverSampler"),
    "RandomUnderSampler": ("imblearn.under_sampling", "RandomUnderSampler"),
    "RepeatedEditedNearestNeighbours": (
        "imblearn.under_sampling",
        "RepeatedEditedNearestNeighbours",
    ),
    "SMOTE": ("imblearn.over_sampling", "SMOTE"),
    "SMOTEENN": ("imblearn.combine", "SMOTEENN"),
    "SMOTEN": ("imblearn.over_sampling", "SMOTEN"),
    "SMOTENC": ("imblearn.over_sampling", "SMOTENC"),
    "SMOTETomek": ("imblearn.combine", "SMOTETomek"),
    "SVMSMOTE": ("imblearn.over_sampling", "SVMSMOTE"),
    "TomekLinks": ("imblearn.under_sampling", "TomekLinks"),
}

__all__ = [
    "ADASYN",
    "AllKNN",
    "BorderlineSMOTE",
    "CondensedNearestNeighbour",
    "EditedNearestNeighbours",
    "InstanceHardnessThreshold",
    "KMeansSMOTE",
    "NearMiss",
    "NeighbourhoodCleaningRule",
    "OneSidedSelection",
    "RandomOverSampler",
    "RandomUnderSampler",
    "RemoveOutliers",
    "RepeatedEditedNearestNeighbours",
    "SMOTE",
    "SMOTEENN",
    "SMOTEN",
    "SMOTENC",
    "SMOTETomek",
    "SVMSMOTE",
    "TomekLinks",
]


def __getattr__(name: str) -> Any:
    if name in _IMBLEARN_EXPORTS:
        module_name, attr = _IMBLEARN_EXPORTS[name]
        try:
            mod = import_module(module_name)
        except ImportError as exc:
            from sklplus._optional import missing_extra_error

            raise missing_extra_error("imbalanced-learn", "imblearn") from exc
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
