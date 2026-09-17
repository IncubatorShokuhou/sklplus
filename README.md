# sklplus

[中文文档](README.zh-CN.md)

sklplus 0.1.0 is a thin umbrella around scikit-learn-style tabular estimators. Most symbols are re-exports (often the same object as upstream). A few adapters and custom preprocessors fill gaps we care about when building `Pipeline`s by hand.

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

### Resampling

Use `ImbPipeline` when a step calls `fit_resample` (e.g. `SMOTE`). Plain `Pipeline` is the sklearn one.

## Examples

Scripts under [`examples/`](examples/):

| Script | Content |
|--------|---------|
| `01_classification_pipeline.py` | `Pipeline` + scaler + RF |
| `02_imbalanced_imbpipeline.py` | `ImbPipeline` + SMOTE + logistic |
| `03_boosting_dual_path.py` | Dual-path identity + small XGB fit |
| `04_preprocessing_p0.py` | P0 custom transformers on a DataFrame |
| `05_anomaly_iforest.py` | Scaler + `IForest` |

```bash
python examples/01_classification_pipeline.py
```

## Custom preprocessors in 0.1.0

These live in `sklplus.preprocessing` (alongside re-exported scalers/encoders):

- `CleanColumnNames`
- `DateFeatureExtractor`
- `RareCategoryGrouper`
- `TargetLabelEncoder`

Anomaly detectors under `sklplus.anomaly` are thin wrappers around pyod (`IForest`, `LOF`, …) so they can sit in a sklearn `Pipeline`.

## Known gaps

Not implemented yet (called P1 in the design notes): `GroupFeatures`, `RemoveMulticollinearity`, `RemoveOutliers`, `IterativeImputerPlus`, `TextEmbedder`; Optuna/skopt search wrappers; broader `check_estimator` coverage; a docs site.

License: TODO (not declared in `pyproject.toml` yet).

Contributing: TODO.
