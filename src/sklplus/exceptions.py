"""Public exception and warning types for sklplus."""

from __future__ import annotations


class SklplusError(Exception):
    """Base exception for sklplus."""


class PipelineConfigurationError(SklplusError, ValueError):
    """Pipeline was constructed with an invalid step configuration."""


class PipelineKindError(PipelineConfigurationError):
    """A fit_resample-only sampler was placed in a sklearn Pipeline."""


class StepConflictError(PipelineConfigurationError):
    """Two steps in the same pipeline are mutually exclusive."""


class ColumnDependencyError(PipelineConfigurationError):
    """A later step names columns that an earlier step would drop."""


class PipelineConflictWarning(UserWarning):
    """Soft / recoverable pipeline configuration issue."""


__all__ = [
    "ColumnDependencyError",
    "PipelineConfigurationError",
    "PipelineConflictWarning",
    "PipelineKindError",
    "SklplusError",
    "StepConflictError",
]
