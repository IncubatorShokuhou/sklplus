"""Pipeline conflict checks: exceptions, rule table, and thin wrappers."""

from __future__ import annotations

import warnings

import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from sklplus.exceptions import (
    ColumnDependencyError,
    PipelineConfigurationError,
    PipelineConflictWarning,
    PipelineKindError,
    SklplusError,
    StepConflictError,
)
from sklplus.impute import IterativeImputer
from sklplus.linear_model import LogisticRegression as PlusLogReg
from sklplus.pipeline import (
    ImbPipeline,
    Pipeline,
    make_pipeline,
    validate_pipeline_steps,
)
from sklplus.preprocessing import (
    DateFeatureExtractor,
    GroupFeatures,
    IterativeImputerPlus,
    RemoveMulticollinearity,
)
from sklplus.sampling import SMOTE, RemoveOutliers, TomekLinks


class _DummyTransformer:
    """Unknown third-party-like estimator: must not be guessed as scaler/imputer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class _DummySampler:
    def fit_resample(self, X, y):
        return X, y


class _DropOriginalColumns:
    """Statically names columns it would drop when drop_original=True."""

    def __init__(self, columns, drop_original=True):
        self.columns = columns
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class _NamedColumns:
    def __init__(self, columns=None, include=None, groups=None):
        self.columns = columns
        self.include = include
        self.groups = groups

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def _clf():
    return LogisticRegression(max_iter=200)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exception_hierarchy():
    assert issubclass(SklplusError, Exception)
    assert issubclass(PipelineConfigurationError, SklplusError)
    assert issubclass(PipelineConfigurationError, ValueError)
    assert issubclass(PipelineKindError, PipelineConfigurationError)
    assert issubclass(StepConflictError, PipelineConfigurationError)
    assert issubclass(ColumnDependencyError, PipelineConfigurationError)
    assert issubclass(PipelineConflictWarning, UserWarning)


def test_pipeline_kind_error_is_value_error():
    with pytest.raises(ValueError):
        raise PipelineKindError("sampler in sklearn pipeline")


# ---------------------------------------------------------------------------
# Rule 1: fit_resample-only samplers in sklearn Pipeline
# ---------------------------------------------------------------------------


def test_smote_in_sklearn_pipeline_raises_kind_error():
    steps = [("sample", SMOTE()), ("clf", _clf())]
    with pytest.raises(PipelineKindError, match="SMOTE"):
        validate_pipeline_steps(steps, kind="sklearn")


def test_remove_outliers_in_sklearn_pipeline_raises_kind_error():
    steps = [("out", RemoveOutliers()), ("clf", _clf())]
    with pytest.raises(PipelineKindError, match="RemoveOutliers"):
        Pipeline(steps)


def test_other_imblearn_sampler_in_sklearn_pipeline_raises():
    steps = [("tl", TomekLinks()), ("clf", _clf())]
    with pytest.raises(PipelineKindError):
        validate_pipeline_steps(steps, kind="sklearn")


def test_dummy_fit_resample_only_in_sklearn_pipeline_raises():
    steps = [("s", _DummySampler()), ("clf", _clf())]
    with pytest.raises(PipelineKindError):
        validate_pipeline_steps(steps, kind="sklearn")


def test_sampler_allowed_in_imbpipeline_kind():
    steps = [("sample", SMOTE()), ("clf", _clf())]
    validate_pipeline_steps(steps, kind="imblearn")
    pipe = ImbPipeline(steps)
    assert pipe.named_steps["sample"].__class__.__name__ == "SMOTE"


def test_pipeline_constructor_rejects_smote():
    with pytest.raises(PipelineKindError):
        Pipeline([("sample", SMOTE()), ("clf", _clf())])


# ---------------------------------------------------------------------------
# Rule 2: consecutive global scalers → warning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first,second",
    [
        (StandardScaler(), MinMaxScaler()),
        (RobustScaler(), Normalizer()),
        (MaxAbsScaler(), StandardScaler()),
    ],
)
def test_consecutive_global_scalers_warn(first, second):
    steps = [("a", first), ("b", second), ("clf", _clf())]
    with pytest.warns(PipelineConflictWarning, match="scaler"):
        validate_pipeline_steps(steps, kind="sklearn")


def test_nonconsecutive_scalers_do_not_warn():
    steps = [
        ("a", StandardScaler()),
        ("mid", PowerTransformer()),
        ("b", MinMaxScaler()),
        ("clf", _clf()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        validate_pipeline_steps(steps, kind="sklearn")


def test_single_scaler_does_not_warn():
    steps = [("scaler", StandardScaler()), ("clf", _clf())]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        Pipeline(steps)


# ---------------------------------------------------------------------------
# Rule 3: RemoveMulticollinearity vs PCA family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pca", [PCA(), IncrementalPCA(), KernelPCA()])
def test_multicollinearity_and_pca_conflict(pca):
    steps = [
        ("rm", RemoveMulticollinearity()),
        ("pca", pca),
        ("clf", _clf()),
    ]
    with pytest.raises(StepConflictError, match="RemoveMulticollinearity"):
        validate_pipeline_steps(steps, kind="sklearn")


def test_pca_without_multicollinearity_ok():
    validate_pipeline_steps(
        [("pca", PCA()), ("clf", _clf())],
        kind="sklearn",
    )


def test_anomaly_pca_is_not_decomposition_pca():
    from sklplus.anomaly import PCA as AnomalyPCA

    steps = [
        ("rm", RemoveMulticollinearity()),
        ("od", AnomalyPCA()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        validate_pipeline_steps(steps, kind="sklearn")


# ---------------------------------------------------------------------------
# Rule 4: consecutive imputers → warning
# ---------------------------------------------------------------------------


def test_consecutive_imputers_warn():
    steps = [
        ("i1", SimpleImputer()),
        ("i2", IterativeImputer()),
        ("clf", _clf()),
    ]
    with pytest.warns(PipelineConflictWarning, match="imputer"):
        validate_pipeline_steps(steps, kind="sklearn")


def test_iterative_imputer_plus_consecutive_warns():
    steps = [
        ("i1", SimpleImputer()),
        ("i2", IterativeImputerPlus()),
        ("clf", _clf()),
    ]
    with pytest.warns(PipelineConflictWarning, match="imputer"):
        Pipeline(steps)


def test_nonconsecutive_imputers_do_not_warn():
    steps = [
        ("i1", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("i2", SimpleImputer()),
        ("clf", _clf()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        validate_pipeline_steps(steps, kind="sklearn")


# ---------------------------------------------------------------------------
# Rule 5: column dependency when drop_original is statically visible
# ---------------------------------------------------------------------------


def test_group_features_drop_original_conflicts_with_later_columns():
    steps = [
        (
            "grp",
            GroupFeatures(groups={"g": ["a", "b"]}, drop_original=True),
        ),
        ("later", _NamedColumns(columns=["a"])),
        ("clf", _clf()),
    ]
    with pytest.raises(ColumnDependencyError, match=r"\ba\b"):
        validate_pipeline_steps(steps, kind="sklearn")


def test_group_features_keep_original_does_not_conflict():
    steps = [
        (
            "grp",
            GroupFeatures(groups={"g": ["a", "b"]}, drop_original=False),
        ),
        ("later", _NamedColumns(columns=["a"])),
        ("clf", _clf()),
    ]
    validate_pipeline_steps(steps, kind="sklearn")


def test_later_include_and_groups_detect_dropped_columns():
    earlier = _DropOriginalColumns(columns=["x", "y"], drop_original=True)
    with pytest.raises(ColumnDependencyError):
        validate_pipeline_steps(
            [
                ("dropper", earlier),
                ("later", _NamedColumns(include=["x"])),
            ],
            kind="sklearn",
        )
    with pytest.raises(ColumnDependencyError):
        validate_pipeline_steps(
            [
                ("dropper", earlier),
                ("later", _NamedColumns(groups={"g": ["y", "z"]})),
            ],
            kind="sklearn",
        )


def test_date_extractor_without_static_columns_is_skipped():
    """DateFeatureExtractor drop set is not known until fit — do not guess."""
    steps = [
        ("dates", DateFeatureExtractor(drop_original=True)),
        ("later", _NamedColumns(columns=["ts"])),
        ("clf", _clf()),
    ]
    validate_pipeline_steps(steps, kind="sklearn")


def test_non_string_columns_attr_is_skipped():
    steps = [
        ("dropper", _DropOriginalColumns(columns=slice(0, 2), drop_original=True)),
        ("later", _NamedColumns(columns=["a"])),
    ]
    validate_pipeline_steps(steps, kind="sklearn")


# ---------------------------------------------------------------------------
# Rule 6: unknown third-party estimators are skipped
# ---------------------------------------------------------------------------


def test_unknown_estimators_are_not_guessed_as_scalers_or_imputers():
    steps = [
        ("u1", _DummyTransformer()),
        ("u2", _DummyTransformer()),
        ("clf", _clf()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        validate_pipeline_steps(steps, kind="sklearn")
        Pipeline(steps)


# ---------------------------------------------------------------------------
# API: check_conflicts, make_pipeline, identity, clone
# ---------------------------------------------------------------------------


def test_check_conflicts_false_skips_errors_and_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        pipe = Pipeline(
            [
                ("a", StandardScaler()),
                ("b", MinMaxScaler()),
                ("sample", SMOTE()),
                ("clf", _clf()),
            ],
            check_conflicts=False,
        )
    assert pipe.check_conflicts is False


def test_check_conflicts_defaults_true():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", _clf())])
    assert pipe.check_conflicts is True


def test_make_pipeline_uses_sklplus_pipeline_and_checks():
    pipe = make_pipeline(StandardScaler(), PlusLogReg(max_iter=200))
    assert type(pipe) is Pipeline
    assert isinstance(pipe, SklearnPipeline)
    with pytest.raises(PipelineKindError):
        make_pipeline(SMOTE(), _clf())


def test_pipeline_is_not_sklearn_identity():
    from imblearn.pipeline import Pipeline as ImblearnPipeline
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.pipeline import make_pipeline as sk_make_pipeline

    assert Pipeline is not SkPipeline
    assert make_pipeline is not sk_make_pipeline
    assert ImbPipeline is not ImblearnPipeline
    assert issubclass(Pipeline, SkPipeline)
    assert issubclass(ImbPipeline, ImblearnPipeline)


def test_validate_pipeline_steps_kind_required():
    with pytest.raises(TypeError):
        validate_pipeline_steps([("clf", _clf())])


def test_validate_pipeline_steps_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        validate_pipeline_steps([("clf", _clf())], kind="caret")


def test_passthrough_and_drop_steps_are_ignored():
    steps = [
        ("scaler", StandardScaler()),
        ("skip", "passthrough"),
        ("other", MinMaxScaler()),
        ("gone", "drop"),
        ("clf", _clf()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", PipelineConflictWarning)
        validate_pipeline_steps(steps, kind="sklearn")


def test_clone_roundtrip_keeps_check_conflicts():
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", _clf())],
        check_conflicts=True,
    )
    cloned = clone(pipe)
    assert isinstance(cloned, Pipeline)
    assert cloned.check_conflicts is True


def test_imbpipeline_check_conflicts_false():
    pipe = ImbPipeline(
        [("sample", SMOTE()), ("clf", _clf())],
        check_conflicts=False,
    )
    assert isinstance(pipe, ImbPipeline)
    assert pipe.check_conflicts is False
