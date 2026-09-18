"""Thin sklearn-compatible adapters around pyod detectors."""

from __future__ import annotations

import inspect
from typing import Any

from sklearn.base import BaseEstimator, OutlierMixin

try:
    import pyod.models.abod as pyod_abod
    import pyod.models.cblof as pyod_cblof
    import pyod.models.cof as pyod_cof
    import pyod.models.hbos as pyod_hbos
    import pyod.models.iforest as pyod_iforest
    import pyod.models.knn as pyod_knn
    import pyod.models.lof as pyod_lof
    import pyod.models.mcd as pyod_mcd
    import pyod.models.ocsvm as pyod_ocsvm
    import pyod.models.pca as pyod_pca
    import pyod.models.sod as pyod_sod
    import pyod.models.sos as pyod_sos
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    raise missing_extra_error("pyod", "anomaly") from exc


def _init_signature(detector_cls: type) -> inspect.Signature:
    sig = inspect.signature(detector_cls.__init__)
    params = [
        p
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    return inspect.Signature(params)


class _PyODAdapter(OutlierMixin, BaseEstimator):
    """Base wrapper: same constructor params as the upstream pyod detector."""

    _detector_cls: type | None = None
    _init_sig: inspect.Signature | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self._init_sig is None:
            raise TypeError(f"{type(self).__name__} is not a concrete adapter")
        bound = self._init_sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for key, value in bound.arguments.items():
            setattr(self, key, value)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        assert self._init_sig is not None
        return {name: getattr(self, name) for name in self._init_sig.parameters}

    def set_params(self, **params: Any):
        for key, value in params.items():
            if self._init_sig is None or key not in self._init_sig.parameters:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}."
                )
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        assert self._detector_cls is not None
        self.detector_ = self._detector_cls(**self.get_params(deep=False))
        self.detector_.fit(X)
        return self

    def predict(self, X):
        return self.detector_.predict(X)

    def decision_function(self, X):
        return self.detector_.decision_function(X)


def _build_adapter(name: str, detector_cls: type) -> type[_PyODAdapter]:
    init_sig = _init_signature(detector_cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _PyODAdapter.__init__(self, *args, **kwargs)

    __init__.__signature__ = init_sig  # type: ignore[attr-defined]

    cls = type(
        name,
        (_PyODAdapter,),
        {
            "_detector_cls": detector_cls,
            "_init_sig": init_sig,
            "__init__": __init__,
            "__module__": __name__,
            "__doc__": f"sklearn-compatible adapter for pyod `{name}`.",
        },
    )
    return cls


ABOD = _build_adapter("ABOD", pyod_abod.ABOD)
CBLOF = _build_adapter("CBLOF", pyod_cblof.CBLOF)
COF = _build_adapter("COF", pyod_cof.COF)
IForest = _build_adapter("IForest", pyod_iforest.IForest)
HBOS = _build_adapter("HBOS", pyod_hbos.HBOS)
KNN = _build_adapter("KNN", pyod_knn.KNN)
LOF = _build_adapter("LOF", pyod_lof.LOF)
OCSVM = _build_adapter("OCSVM", pyod_ocsvm.OCSVM)
PCA = _build_adapter("PCA", pyod_pca.PCA)
MCD = _build_adapter("MCD", pyod_mcd.MCD)
SOD = _build_adapter("SOD", pyod_sod.SOD)
SOS = _build_adapter("SOS", pyod_sos.SOS)

__all__ = [
    "ABOD",
    "CBLOF",
    "COF",
    "IForest",
    "HBOS",
    "KNN",
    "LOF",
    "OCSVM",
    "PCA",
    "MCD",
    "SOD",
    "SOS",
]
