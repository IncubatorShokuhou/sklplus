"""Thin sklearn Pipeline wrapper with optional conflict checks."""

from __future__ import annotations

import inspect

from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.pipeline import _name_estimators

from sklplus.pipeline.validate import validate_pipeline_steps

_SKLEARN_HAS_TRANSFORM_INPUT = (
    "transform_input" in inspect.signature(_SklearnPipeline.__init__).parameters
)


def _forward_pipeline_kwargs(
    *,
    memory,
    verbose,
    transform_input,
    parent_accepts_transform_input: bool,
) -> dict:
    kwargs = {"memory": memory, "verbose": verbose}
    if parent_accepts_transform_input:
        kwargs["transform_input"] = transform_input
    return kwargs


class Pipeline(_SklearnPipeline):
    """sklearn Pipeline with optional step conflict checks.

    Parameters
    ----------
    steps :
        List of ``(name, transform)`` tuples (final estimator last).
    check_conflicts :
        If True (default), run :func:`validate_pipeline_steps` with
        ``kind="sklearn"`` before construction.
    transform_input, memory, verbose :
        Forwarded to ``sklearn.pipeline.Pipeline`` when supported.
    """

    def __init__(
        self,
        steps,
        *,
        check_conflicts=True,
        transform_input=None,
        memory=None,
        verbose=False,
    ):
        self.check_conflicts = check_conflicts
        if check_conflicts:
            validate_pipeline_steps(steps, kind="sklearn", stacklevel=3)
        if not _SKLEARN_HAS_TRANSFORM_INPUT:
            self.transform_input = transform_input
        super().__init__(
            steps,
            **_forward_pipeline_kwargs(
                memory=memory,
                verbose=verbose,
                transform_input=transform_input,
                parent_accepts_transform_input=_SKLEARN_HAS_TRANSFORM_INPUT,
            ),
        )


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
    "Pipeline",
    "make_pipeline",
]
