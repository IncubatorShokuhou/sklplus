"""Pipeline helpers. ``ImbPipeline`` needs extra ``imblearn``."""

from __future__ import annotations

from typing import Any

from sklplus.pipeline._pipeline import Pipeline, make_pipeline
from sklplus.pipeline.validate import validate_pipeline_steps

__all__ = [
    "ImbPipeline",
    "Pipeline",
    "make_pipeline",
    "validate_pipeline_steps",
]


def __getattr__(name: str) -> Any:
    if name == "ImbPipeline":
        from sklplus.pipeline._imb_pipeline import ImbPipeline

        globals()["ImbPipeline"] = ImbPipeline
        return ImbPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
