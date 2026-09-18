"""imblearn Pipeline wrapper (optional extra: imblearn)."""

from __future__ import annotations

import inspect

try:
    from imblearn.pipeline import Pipeline as _ImblearnPipeline
except ImportError as exc:
    from sklplus._optional import missing_extra_error

    raise missing_extra_error("imbalanced-learn", "imblearn") from exc

from sklplus.pipeline._pipeline import _forward_pipeline_kwargs
from sklplus.pipeline.validate import validate_pipeline_steps

_IMBLEARN_HAS_TRANSFORM_INPUT = (
    "transform_input" in inspect.signature(_ImblearnPipeline.__init__).parameters
)


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
    transform_input, memory, verbose :
        Forwarded to ``imblearn.pipeline.Pipeline`` when supported.
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
            validate_pipeline_steps(steps, kind="imblearn", stacklevel=3)
        if not _IMBLEARN_HAS_TRANSFORM_INPUT:
            self.transform_input = transform_input
        super().__init__(
            steps,
            **_forward_pipeline_kwargs(
                memory=memory,
                verbose=verbose,
                transform_input=transform_input,
                parent_accepts_transform_input=_IMBLEARN_HAS_TRANSFORM_INPUT,
            ),
        )


__all__ = ["ImbPipeline"]
