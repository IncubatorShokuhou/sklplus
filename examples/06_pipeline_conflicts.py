"""Pipeline conflict checks: print hard errors and soft warnings."""

from __future__ import annotations

import warnings

from sklplus.decomposition import PCA
from sklplus.exceptions import (
    ColumnDependencyError,
    PipelineConfigurationError,
    PipelineConflictWarning,
    PipelineKindError,
    StepConflictError,
)
from sklplus.impute import IterativeImputer, SimpleImputer
from sklplus.linear_model import LogisticRegression
from sklplus.pipeline import Pipeline
from sklplus.preprocessing import (
    GroupFeatures,
    MinMaxScaler,
    RemoveMulticollinearity,
    StandardScaler,
    TextEmbedder,
)
from sklplus.sampling import SMOTE


def _clf():
    return LogisticRegression(max_iter=200)


def _run_error(label: str, expected: type[BaseException], build) -> bool:
    try:
        build()
    except expected as exc:
        print(f"{label}\n  {type(exc).__name__}: {exc}")
        return True
    except PipelineConfigurationError as exc:
        print(f"{label}\n  FAILED — unexpected {type(exc).__name__}: {exc}")
        return False
    print(f"{label}\n  FAILED — expected {expected.__name__}")
    return False


def _run_warning(label: str, build) -> bool:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", PipelineConflictWarning)
        try:
            build()
        except PipelineConfigurationError as exc:
            print(f"{label}\n  FAILED — unexpected {type(exc).__name__}: {exc}")
            return False
    matches = [w for w in recorded if issubclass(w.category, PipelineConflictWarning)]
    if not matches:
        print(f"{label}\n  FAILED — expected PipelineConflictWarning")
        return False
    msg = matches[0]
    print(f"{label}\n  {msg.category.__name__}: {msg.message}")
    return True


def main() -> int:
    ok = True
    ok &= _run_error(
        "[1] PipelineKindError — SMOTE 放进 sklearn Pipeline",
        PipelineKindError,
        lambda: Pipeline([("sample", SMOTE()), ("clf", _clf())]),
    )
    ok &= _run_warning(
        "[2] PipelineConflictWarning — 连续 StandardScaler + MinMaxScaler",
        lambda: Pipeline(
            [
                ("std", StandardScaler()),
                ("minmax", MinMaxScaler()),
                ("clf", _clf()),
            ]
        ),
    )
    ok &= _run_error(
        "[3] StepConflictError — RemoveMulticollinearity + PCA",
        StepConflictError,
        lambda: Pipeline(
            [
                ("rm", RemoveMulticollinearity()),
                ("pca", PCA()),
                ("clf", _clf()),
            ]
        ),
    )
    ok &= _run_warning(
        "[4] PipelineConflictWarning — 连续 imputers",
        lambda: Pipeline(
            [
                ("i1", SimpleImputer()),
                ("i2", IterativeImputer()),
                ("clf", _clf()),
            ]
        ),
    )
    ok &= _run_error(
        "[5] ColumnDependencyError — GroupFeatures(drop_original=True) "
        "后再引用被丢列",
        ColumnDependencyError,
        lambda: Pipeline(
            [
                (
                    "grp",
                    GroupFeatures(groups={"g": ["a", "b"]}, drop_original=True),
                ),
                ("later", TextEmbedder(columns=["a"])),
                ("clf", _clf()),
            ]
        ),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
