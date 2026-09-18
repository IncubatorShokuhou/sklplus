"""Helpers for optional dependency extras."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def missing_extra_error(dependency: str, extra: str) -> ImportError:
    """Return a clear ImportError naming the pip extra to install."""
    return ImportError(
        f"{dependency} is required for this feature. "
        f'Install with: pip install "sklplus[{extra}]"'
    )


def require_extra(module_name: str, extra: str, *, dependency: str | None = None) -> Any:
    """Import ``module_name`` or raise ImportError naming ``extra``."""
    dep = dependency or module_name.split(".", 1)[0]
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise missing_extra_error(dep, extra) from exc
