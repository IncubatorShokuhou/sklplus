"""Smoke-run example scripts via their main() entrypoints."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main"), f"{path.name} must define main()"
    return mod.main


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=[p.name for p in EXAMPLE_SCRIPTS])
def test_example_main_returns_zero(script: Path):
    main = _load_main(script)
    assert main() == 0
