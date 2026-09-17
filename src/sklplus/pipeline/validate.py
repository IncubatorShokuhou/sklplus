"""Pure-function pipeline step conflict checks and the v0.1 rule table."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from sklplus.exceptions import (
    ColumnDependencyError,
    PipelineConflictWarning,
    PipelineKindError,
    StepConflictError,
)
from sklplus.preprocessing import IterativeImputerPlus, RemoveMulticollinearity

_ALLOWED_KINDS = frozenset({"sklearn", "imblearn"})
_SKIPPED_ESTIMATORS = {None, "passthrough", "drop"}
_GLOBAL_SCALERS = (StandardScaler, MinMaxScaler, RobustScaler, Normalizer, MaxAbsScaler)
_IMPUTERS = (SimpleImputer, IterativeImputer, IterativeImputerPlus)
_PCA_FAMILY = (PCA, IncrementalPCA, KernelPCA)
_COLUMN_ATTRS = ("columns", "include", "groups")


@dataclass(frozen=True)
class PipelineIssue:
    """One finding from a pipeline rule."""

    severity: str
    message: str
    exc_type: type[BaseException] | None = None


@dataclass(frozen=True)
class PipelineRule:
    """Extensible rule: ``check(steps, *, kind)`` returns issues."""

    name: str
    check: Callable[..., list[PipelineIssue]]


def _iter_named_steps(steps) -> list[tuple[str, object]]:
    if steps is None:
        return []
    try:
        items = list(steps)
    except TypeError:
        return []
    named: list[tuple[str, object]] = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            named.append((str(item[0]), item[1]))
    return named


def _is_skipped(est) -> bool:
    return est in _SKIPPED_ESTIMATORS


def _cls_name(est) -> str:
    return type(est).__name__


def _has_callable(est, name: str) -> bool:
    return callable(getattr(est, name, None))


def _is_fit_resample_only(est) -> bool:
    return _has_callable(est, "fit_resample") and not _has_callable(est, "transform")


def _static_names(value) -> set[str] | None:
    """Column names if statically visible; ``None`` means skip (do not guess)."""
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        names: set[str] = set()
        visible = False
        for inner in value.values():
            part = _static_names(inner)
            if part is None:
                continue
            visible = True
            names.update(part)
        if visible or len(value) == 0:
            return names
        return None
    if isinstance(value, (list, tuple, set)):
        if value and all(isinstance(item, str) for item in value):
            return set(value)
        if not value:
            return set()
        return None
    return None


def _named_columns(est) -> set[str]:
    names: set[str] = set()
    for attr in _COLUMN_ATTRS:
        if not hasattr(est, attr):
            continue
        part = _static_names(getattr(est, attr))
        if part:
            names.update(part)
    return names


def _check_fit_resample_kind(steps, *, kind: str) -> list[PipelineIssue]:
    if kind != "sklearn":
        return []
    issues: list[PipelineIssue] = []
    for name, est in _iter_named_steps(steps):
        if _is_skipped(est) or not _is_fit_resample_only(est):
            continue
        cls = _cls_name(est)
        issues.append(
            PipelineIssue(
                severity="error",
                message=(
                    f"fit_resample-only sampler {cls!r} in step {name!r} cannot "
                    "be used in a sklearn Pipeline; use ImbPipeline"
                ),
                exc_type=PipelineKindError,
            )
        )
    return issues


def _check_consecutive_scalers(steps, *, kind: str) -> list[PipelineIssue]:
    del kind
    items = _iter_named_steps(steps)
    issues: list[PipelineIssue] = []
    for (n1, e1), (n2, e2) in pairwise(items):
        if _is_skipped(e1) or _is_skipped(e2):
            continue
        if isinstance(e1, _GLOBAL_SCALERS) and isinstance(e2, _GLOBAL_SCALERS):
            issues.append(
                PipelineIssue(
                    severity="warning",
                    message=(
                        f"consecutive global scalers: {n1!r} ({_cls_name(e1)}) "
                        f"followed by {n2!r} ({_cls_name(e2)})"
                    ),
                )
            )
    return issues


def _check_multicollinearity_pca(steps, *, kind: str) -> list[PipelineIssue]:
    del kind
    rm_steps: list[tuple[str, object]] = []
    pca_steps: list[tuple[str, object]] = []
    for name, est in _iter_named_steps(steps):
        if _is_skipped(est):
            continue
        if isinstance(est, RemoveMulticollinearity):
            rm_steps.append((name, est))
        if isinstance(est, _PCA_FAMILY):
            pca_steps.append((name, est))
    if not rm_steps or not pca_steps:
        return []
    rm_name, _ = rm_steps[0]
    pca_name, pca_est = pca_steps[0]
    return [
        PipelineIssue(
            severity="error",
            message=(
                f"RemoveMulticollinearity step {rm_name!r} conflicts with "
                f"{_cls_name(pca_est)} step {pca_name!r} in the same pipeline"
            ),
            exc_type=StepConflictError,
        )
    ]


def _check_consecutive_imputers(steps, *, kind: str) -> list[PipelineIssue]:
    del kind
    items = _iter_named_steps(steps)
    issues: list[PipelineIssue] = []
    for (n1, e1), (n2, e2) in pairwise(items):
        if _is_skipped(e1) or _is_skipped(e2):
            continue
        if isinstance(e1, _IMPUTERS) and isinstance(e2, _IMPUTERS):
            issues.append(
                PipelineIssue(
                    severity="warning",
                    message=(
                        f"consecutive imputers: {n1!r} ({_cls_name(e1)}) "
                        f"followed by {n2!r} ({_cls_name(e2)})"
                    ),
                )
            )
    return issues


def _check_column_dependency(steps, *, kind: str) -> list[PipelineIssue]:
    del kind
    dropped: dict[str, str] = {}
    issues: list[PipelineIssue] = []
    for name, est in _iter_named_steps(steps):
        if _is_skipped(est):
            continue
        needed = _named_columns(est)
        overlap = sorted(col for col in needed if col in dropped)
        if overlap:
            earlier = dropped[overlap[0]]
            issues.append(
                PipelineIssue(
                    severity="error",
                    message=(
                        f"step {name!r} names column(s) {overlap} that earlier "
                        f"step {earlier!r} would drop (drop_original=True)"
                    ),
                    exc_type=ColumnDependencyError,
                )
            )
        if getattr(est, "drop_original", False):
            for col in _named_columns(est):
                dropped.setdefault(col, name)
    return issues


PIPELINE_RULES: list[PipelineRule] = [
    PipelineRule("fit_resample_kind", _check_fit_resample_kind),
    PipelineRule("consecutive_scalers", _check_consecutive_scalers),
    PipelineRule("multicollinearity_pca", _check_multicollinearity_pca),
    PipelineRule("consecutive_imputers", _check_consecutive_imputers),
    PipelineRule("column_dependency", _check_column_dependency),
]


def validate_pipeline_steps(steps: Sequence, *, kind: str) -> None:
    """Run the v0.1 conflict rules on ``steps``.

    Parameters
    ----------
    steps :
        Sequence of ``(name, estimator)`` pairs, same as sklearn Pipeline.
    kind :
        ``"sklearn"`` or ``"imblearn"``. Sampler kind-errors only apply to
        ``"sklearn"``.

    Raises
    ------
    PipelineKindError, StepConflictError, ColumnDependencyError
        Hard configuration errors.
    ValueError
        If ``kind`` is not a supported pipeline kind.

    Warns
    -----
    PipelineConflictWarning
        Soft issues such as consecutive global scalers or imputers.
    """
    if kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"kind must be 'sklearn' or 'imblearn', got {kind!r}"
        )
    issues: list[PipelineIssue] = []
    for rule in PIPELINE_RULES:
        issues.extend(rule.check(steps, kind=kind))
    for issue in issues:
        if issue.severity == "warning":
            warnings.warn(issue.message, PipelineConflictWarning, stacklevel=2)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        exc_type = errors[0].exc_type or PipelineKindError
        raise exc_type(errors[0].message)


__all__ = [
    "PIPELINE_RULES",
    "PipelineIssue",
    "PipelineRule",
    "validate_pipeline_steps",
]
