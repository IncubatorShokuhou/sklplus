"""Smoke-run example scripts via their main() entrypoints."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
ROOT = EXAMPLES_DIR.parent
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))
CONFLICT_EXAMPLE = EXAMPLES_DIR / "06_pipeline_conflicts.py"


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


def test_pipeline_conflicts_example_prints_conflict_messages():
    assert CONFLICT_EXAMPLE.is_file(), "examples/06_pipeline_conflicts.py must exist"
    proc = subprocess.run(
        [sys.executable, str(CONFLICT_EXAMPLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined
    for needle in (
        "PipelineKindError",
        "PipelineConflictWarning",
        "StepConflictError",
        "ColumnDependencyError",
    ):
        assert needle in combined, f"missing {needle} in:\n{combined}"
    for fragment in (
        "SMOTE",
        "consecutive global scalers",
        "RemoveMulticollinearity",
        "consecutive imputers",
        "column(s) ['a']",
    ):
        assert fragment in combined, f"missing {fragment} in:\n{combined}"
