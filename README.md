# sklplus

[中文文档](README.zh-CN.md) · [中文项目页](https://incubatorshokuhou.github.io/sklplus/) · [Wiki](https://github.com/IncubatorShokuhou/sklplus/wiki)

sklplus 0.1.2 is a thin umbrella around scikit-learn-style tabular estimators. Most symbols are re-exports (often the same object as upstream). A few adapters and custom preprocessors fill gaps we care about when building `Pipeline`s by hand.

It is for people who want `from sklplus.… import …` and standard `fit` / `transform` / `predict` composition. It is not AutoML: there is no `get_model`, no `setup()`, and no string-ID model factory.

**Fits:** classification, regression, clustering, and anomaly detection on tabular data, with preprocess / sampling / selection / search / metrics helpers re-exported under sklearn-like paths.

**Does not fit (v0.1):** time-series or NLP specialist modules. If you need PyCaret-style experiment sessions, use something else.

## Requirements

- Python `>=3.10`
- `scikit-learn>=1.3`
- Default install also pulls: `numpy`, `pandas`, `scipy`, `joblib`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `pyod`, `kmodes`, `category-encoders`, `feature-engine`

## Install

PyPI / import / GitHub repo name: **`sklplus`** (https://github.com/IncubatorShokuhou/sklplus).

```bash
pip install sklplus
```

From a clone (editable, with test tools):

```bash
pip install -e ".[dev]"
```

`[dev]` adds `pytest>=7` and `ruff`.

## Minimal usage

```python
from sklplus.linear_model import LogisticRegression, Ridge
from sklplus.ensemble import RandomForestClassifier, XGBClassifier
from sklplus.preprocessing import (
    StandardScaler,
    CleanColumnNames,
    RareCategoryGrouper,
    DateFeatureExtractor,
    TargetLabelEncoder,
)
from sklplus.sampling import SMOTE
from sklplus.anomaly import IForest
from sklplus.pipeline import Pipeline, ImbPipeline
from sklplus.compose import ColumnTransformer
```

Re-exports aim to be identity with upstream when possible, e.g. `sklplus.ensemble.RandomForestClassifier is sklearn.ensemble.RandomForestClassifier`.

### Boosting import paths

`XGBClassifier` / `LGBMClassifier` / `CatBoostClassifier` (and the regressors) are exported from both `sklplus.ensemble` and `sklplus.xgboost` / `lightgbm` / `catboost`. They are the same class object:

```python
from sklplus.ensemble import XGBClassifier as A
from sklplus.xgboost import XGBClassifier as B
assert A is B
```

### Task hubs and ensemble technique slices

sklearn-style modules (`linear_model`, `ensemble`, `xgboost`, …) stay put. Two extra hubs re-export the same objects by task:

```python
from sklplus.classification import LogisticRegression, XGBClassifier
from sklplus.regression import Ridge, XGBRegressor
from sklplus.linear_model import LogisticRegression as LM
from sklplus.xgboost import XGBClassifier as XGB
assert LogisticRegression is LM
assert XGBClassifier is XGB
```

`sklplus.ensemble` is still the full umbrella. Technique submodules are slices, not moves:

- `sklplus.ensemble.bagging` — `BaggingClassifier` / `BaggingRegressor` only (RF / ExtraTrees stay on the umbrella)
- `sklplus.ensemble.boosting` — AdaBoost, GradientBoosting, XGB, LGBM, CatBoost
- `sklplus.ensemble.stacking` — `StackingClassifier` / `StackingRegressor`
- `sklplus.ensemble.voting` — `VotingClassifier` / `VotingRegressor`

```python
from sklplus.ensemble.boosting import XGBClassifier as A
from sklplus.xgboost import XGBClassifier as B
from sklplus.ensemble import XGBClassifier as C
assert A is B is C
```

There is no top-level `sklplus.stacking`, and no `get_model` / factory.

### Resampling and pipeline checks

Use `ImbPipeline` when a step calls `fit_resample` (e.g. `SMOTE`, `RemoveOutliers`).

`sklplus.pipeline.Pipeline` is a thin subclass of sklearn's Pipeline, so `sklplus.pipeline.Pipeline is sklearn.pipeline.Pipeline` is False. Construction defaults to `check_conflicts=True`. `make_pipeline` builds this wrapper. Disable with `check_conflicts=False`, or call `validate_pipeline_steps(steps, *, kind)` directly.

## Examples

Scripts under [`examples/`](examples/):

| Script | Content |
|--------|---------|
| `01_classification_pipeline.py` | `Pipeline` + scaler + RF |
| `02_imbalanced_imbpipeline.py` | `ImbPipeline` + SMOTE + logistic |
| `03_boosting_dual_path.py` | Dual-path identity + small XGB fit |
| `04_preprocessing_p0.py` | P0 custom transformers on a DataFrame |
| `05_anomaly_iforest.py` | Scaler + `IForest` |
| `06_pipeline_conflicts.py` | Pipeline conflict checks: errors and warnings |

```bash
python examples/01_classification_pipeline.py
```

## Custom preprocessors in 0.1.2

These live in `sklplus.preprocessing` (alongside re-exported scalers/encoders):

**P0**

- `CleanColumnNames`
- `DateFeatureExtractor`
- `RareCategoryGrouper`
- `TargetLabelEncoder`

**P1**

- `GroupFeatures` — row-wise aggregates over named column groups
- `RemoveMulticollinearity` — drop highly correlated numeric features (optional prefer-by-`y`)
- `IterativeImputerPlus` — `IterativeImputer` for numeric + most-frequent for categoricals
- `TextEmbedder` — BoW / TF-IDF on text columns, concatenated back

**Sampling (not preprocessing):** `RemoveOutliers` in `sklplus.sampling` is an imblearn-style sampler (`fit_resample`) for train-only outlier row drops — use with `ImbPipeline`.

Anomaly detectors under `sklplus.anomaly` are thin wrappers around pyod (`IForest`, `LOF`, …) so they can sit in a sklearn `Pipeline`.

## Known gaps

Still open: Optuna/skopt search wrappers; broader `check_estimator` coverage. P1 custom preprocessors / `RemoveOutliers` sampler are implemented. Pipeline conflict checks and classification/regression task hubs are in.

License: MIT (see `LICENSE`).

Contributing: see [`CONTRIBUTING.md`](CONTRIBUTING.md).
