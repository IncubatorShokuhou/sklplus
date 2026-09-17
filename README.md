# sklearnplus

[中文文档 / Chinese README](README.zh-CN.md)

**Big sklearn** for tabular ML: re-export familiar estimators under sklearn-style
paths, plus thin adapters and a few custom preprocessors where the ecosystem has
gaps.

This is a **component library**, not AutoML. There is **no** `get_model`,
`setup`, or string-ID model factory — import classes and compose `Pipeline`s
yourself.

Default install pulls boosting (xgboost / lightgbm / catboost), imbalanced-learn,
pyod, kmodes, category-encoders, and feature-engine so listed imports work out of
the box.

## Install

```bash
pip install -e ".[dev]"
```

## Philosophy

| Do | Don't |
|----|-------|
| `from sklearnplus.ensemble import RandomForestClassifier` | `get_model("rf")` |
| Compose `Pipeline` / `ImbPipeline` | Global `setup()` session state |
| Same objects as upstream when re-exporting | Opaque wrappers that change algorithm semantics |

## Boosting dual-path

XGBoost / LightGBM / CatBoost are available from both `sklearnplus.ensemble`
and dedicated submodules — **the same class object**:

```python
from sklearnplus.ensemble import XGBClassifier as A
from sklearnplus.xgboost import XGBClassifier as B
assert A is B
```

## Example imports

```python
from sklearnplus.linear_model import LogisticRegression, Ridge
from sklearnplus.ensemble import RandomForestClassifier, XGBClassifier
from sklearnplus.preprocessing import (
    StandardScaler,
    CleanColumnNames,
    RareCategoryGrouper,
    DateFeatureExtractor,
    TargetLabelEncoder,
)
from sklearnplus.sampling import SMOTE
from sklearnplus.anomaly import IForest
from sklearnplus.pipeline import Pipeline, ImbPipeline
from sklearnplus.compose import ColumnTransformer
```

## Runnable examples

See [`examples/`](examples/):

| Script | What it shows |
|--------|----------------|
| `01_classification_pipeline.py` | `Pipeline` + scaler + RF |
| `02_imbalanced_imbpipeline.py` | `ImbPipeline` + SMOTE + logistic |
| `03_boosting_dual_path.py` | Dual-path identity + tiny XGB fit |
| `04_preprocessing_p0.py` | P0 custom transformers on a DataFrame |
| `05_anomaly_iforest.py` | Scaler + `IForest` anomaly pipeline |

```bash
python examples/01_classification_pipeline.py
```

## Scope (v0.1)

- Classification, regression, clustering, anomaly detection
- Preprocess / sample / select / search / metrics helpers
- P0 customs: `CleanColumnNames`, `DateFeatureExtractor`, `RareCategoryGrouper`, `TargetLabelEncoder`

**Not in v0.1:** time-series or NLP specialist modules.

## Deferred (P1)

`GroupFeatures`, `RemoveMulticollinearity`, `RemoveOutliers`,
`IterativeImputerPlus`, `TextEmbedder`; Optuna/skopt wrappers; fuller
`check_estimator` coverage; docs site.
