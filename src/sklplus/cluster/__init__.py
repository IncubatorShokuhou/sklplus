"""Clustering estimators. ``KModes`` needs extra ``extra``."""

from __future__ import annotations

from typing import Any

from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    OPTICS,
    SpectralClustering,
)

__all__ = [
    "AffinityPropagation",
    "AgglomerativeClustering",
    "Birch",
    "DBSCAN",
    "KMeans",
    "KModes",
    "MeanShift",
    "OPTICS",
    "SpectralClustering",
]


def __getattr__(name: str) -> Any:
    if name == "KModes":
        try:
            from kmodes.kmodes import KModes
        except ImportError as exc:
            from sklplus._optional import missing_extra_error

            raise missing_extra_error("kmodes", "extra") from exc
        globals()["KModes"] = KModes
        return KModes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
