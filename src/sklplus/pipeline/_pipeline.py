"""Thin sklearn / imblearn Pipeline wrappers with optional conflict checks."""

from __future__ import annotations

from imblearn.pipeline import Pipeline as _ImblearnPipeline
from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.pipeline import _name_estimators

from sklplus.pipeline.validate import validate_pipeline_steps


class Pipeline(_SklearnPipeline):
    """sklearn Pipeline with optional step conflict checks.

    Parameters
    ----------
    steps :
        List of ``(name, transform)`` tuples (final estimator last).
    check_conflicts :
        If True (default), run :func:`validate_pipeline_steps` with
        ``kind="sklearn"`` before construction.
    **kwargs :
        Forwarded to ``sklearn.pipeline.Pipeline`` (``memory``, ``verbose``,
        ``transform_input``, …).
    """

    def __init__(self, steps, *, check_conflicts=True, **kwargs):
        self.check_conflicts = check_conflicts
        if check_conflicts:
            validate_pipeline_steps(steps, kind="sklearn")
        super().__init__(steps, **kwargs)


class ImbPipeline(_ImblearnPipeline):
    """imblearn Pipeline with optional step conflict checks.

    Parameters
    ----------
    steps :
        List of ``(name, transform)`` tuples (final estimator last).
        ``fit_resample`` samplers are allowed.
    check_conflicts :
        If True (default), run :func:`validate_pipeline_steps` with
        ``kind="imblearn"`` before construction.
    **kwargs :
        Forwarded to ``imblearn.pipeline.Pipeline``.
    """

    def __init__(self, steps, *, check_conflicts=True, **kwargs):
        self.check_conflicts = check_conflicts
        if check_conflicts:
            validate_pipeline_steps(steps, kind="imblearn")
        super().__init__(steps, **kwargs)


def make_pipeline(
    *steps,
    memory=None,
    transform_input=None,
    verbose=False,
    check_conflicts=True,
):
    """Construct a :class:`Pipeline` from the given estimators.

    Same naming behavior as ``sklearn.pipeline.make_pipeline``, but the
    result is :class:`sklplus.pipeline.Pipeline` with conflict checks on
    by default.
    """
    return Pipeline(
        _name_estimators(steps),
        memory=memory,
        transform_input=transform_input,
        verbose=verbose,
        check_conflicts=check_conflicts,
    )


__all__ = [
    "ImbPipeline",
    "Pipeline",
    "make_pipeline",
]
