"""Anomaly detectors (optional extra: anomaly). Thin pyod adapters."""

from __future__ import annotations

try:
    from sklplus.anomaly._adapters import (
        ABOD,
        CBLOF,
        COF,
        HBOS,
        IForest,
        KNN,
        LOF,
        MCD,
        OCSVM,
        PCA,
        SOD,
        SOS,
    )
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    # _adapters imports pyod at module level
    if "pyod" in str(exc).lower() or getattr(exc, "name", None) in {
        "pyod",
        "pyod.models",
    }:
        raise missing_extra_error("pyod", "anomaly") from exc
    # Already a clear sklplus missing-extra error, or unexpected
    if "sklplus[" in str(exc):
        raise
    raise missing_extra_error("pyod", "anomaly") from exc

__all__ = [
    "ABOD",
    "CBLOF",
    "COF",
    "HBOS",
    "IForest",
    "KNN",
    "LOF",
    "MCD",
    "OCSVM",
    "PCA",
    "SOD",
    "SOS",
]
