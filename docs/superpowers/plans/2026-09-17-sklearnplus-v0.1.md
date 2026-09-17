# sklearnplus v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `sklearnplus` as a big-sklearn umbrella package: re-export tabular estimators (sklearn + boosting + imblearn + pyod + kmodes), add P0 custom preprocessors, dual-path boosting imports, all-in-one deps, and CI coverage so nothing on the v0.1 checklist is missing.

**Architecture:** Thin re-export modules mirror sklearn paths; boosting also lives under `sklearnplus.xgboost|lightgbm|catboost`. Custom transformers only for P0 gaps. `_coverage.PUBLIC_SYMBOLS` drives import CI. No `get_model`, no AutoML session API.

**Tech Stack:** Python ≥3.10, scikit-learn, pandas, numpy, xgboost, lightgbm, catboost, imbalanced-learn, pyod, kmodes, category-encoders, feature-engine, pytest, hatchling/setuptools.

**Spec:** `docs/superpowers/specs/2026-09-17-sklearnplus-design.md`

## Global Constraints

- Package name / import root: `sklearnplus` (not `sklearn-plus`).
- No public `get_model` / PyCaret string-ID factory.
- Default install must include all model deps so listed imports work out of the box.
- Scope: classification, regression, clustering, anomaly + preprocess/sampling — **no** time series / NLP modules.
- Boosting: importable from both `sklearnplus.ensemble` and dedicated submodules (same class object).
- Prefer re-export identity (`X is sklearn.X`) over wrappers unless adaptation is required (pyod).
- TDD: failing test → implement → pass → commit per task.
- Commit messages: conventional (`feat:`, `test:`, `chore:`).

---

## File map (v0.1)

```
pyproject.toml
README.md
src/sklearnplus/__init__.py
src/sklearnplus/_coverage.py
src/sklearnplus/linear_model/__init__.py
src/sklearnplus/tree/__init__.py
src/sklearnplus/neighbors/__init__.py
src/sklearnplus/svm/__init__.py
src/sklearnplus/naive_bayes/__init__.py
src/sklearnplus/discriminant_analysis/__init__.py
src/sklearnplus/neural_network/__init__.py
src/sklearnplus/gaussian_process/__init__.py
src/sklearnplus/ensemble/__init__.py
src/sklearnplus/dummy/__init__.py
src/sklearnplus/kernel_ridge/__init__.py
src/sklearnplus/cluster/__init__.py
src/sklearnplus/anomaly/__init__.py
src/sklearnplus/anomaly/_adapters.py
src/sklearnplus/decomposition/__init__.py
src/sklearnplus/preprocessing/__init__.py
src/sklearnplus/preprocessing/_clean_column_names.py
src/sklearnplus/preprocessing/_date_features.py
src/sklearnplus/preprocessing/_rare_category.py
src/sklearnplus/preprocessing/_target_label.py
src/sklearnplus/impute/__init__.py
src/sklearnplus/feature_selection/__init__.py
src/sklearnplus/compose/__init__.py
src/sklearnplus/pipeline/__init__.py
src/sklearnplus/model_selection/__init__.py
src/sklearnplus/metrics/__init__.py
src/sklearnplus/sampling/__init__.py
src/sklearnplus/xgboost/__init__.py
src/sklearnplus/lightgbm/__init__.py
src/sklearnplus/catboost/__init__.py
tests/test_coverage_imports.py
tests/test_reexport_identity.py
tests/test_boosting_dual_path.py
tests/test_preprocessing_p0.py
tests/test_anomaly_adapter.py
tests/test_smoke_pipelines.py
```

---

### Task 1: Project scaffold + coverage skeleton

**Files:**
- Create: `pyproject.toml`, `README.md`, `src/sklearnplus/__init__.py`, `src/sklearnplus/_coverage.py`, `tests/test_coverage_imports.py`

**Interfaces:**
- Produces: `sklearnplus.__version__` str; `sklearnplus._coverage.PUBLIC_SYMBOLS: list[tuple[str, str]]` as `(module_path, attr_name)` e.g. `("sklearnplus.linear_model", "LogisticRegression")`

- [ ] **Step 1: Write failing coverage import test**

```python
# tests/test_coverage_imports.py
import importlib
import pytest
from sklearnplus._coverage import PUBLIC_SYMBOLS

@pytest.mark.parametrize("module_path,attr", PUBLIC_SYMBOLS)
def test_public_symbol_importable(module_path, attr):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} missing {attr}"
```

- [ ] **Step 2: Run test — expect fail (no package / empty or missing modules)**

Run: `pytest tests/test_coverage_imports.py -q`  
Expected: FAIL (collection or import error)

- [ ] **Step 3: Minimal scaffold**

`pyproject.toml` (hatchling, `src/` layout):

```toml
[project]
name = "sklearnplus"
version = "0.1.0"
description = "Big sklearn: re-exports and light wrappers for tabular ML pipelines"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "scikit-learn>=1.3",
  "numpy",
  "pandas",
  "scipy",
  "joblib",
  "xgboost",
  "lightgbm",
  "catboost",
  "imbalanced-learn",
  "pyod",
  "kmodes",
  "category-encoders",
  "feature-engine",
]

[project.optional-dependencies]
dev = ["pytest>=7", "ruff"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sklearnplus"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/sklearnplus/__init__.py
__version__ = "0.1.0"
```

```python
# src/sklearnplus/_coverage.py
"""CI-only public symbol checklist. Not a user-facing registry API."""
PUBLIC_SYMBOLS: list[tuple[str, str]] = [
    # filled in later tasks — start with version probe via package exists
]
```

Keep `PUBLIC_SYMBOLS = []` initially so the parametrize test collects 0 cases (or add one sentinel after linear_model exists). Prefer empty list + separate `test_package_version` :

```python
def test_version():
    import sklearnplus
    assert sklearnplus.__version__ == "0.1.0"
```

- [ ] **Step 4: `pip install -e ".[dev]"` and pass version test**

Run: `pip install -e ".[dev]" && pytest tests/test_coverage_imports.py::test_version -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/sklearnplus/__init__.py src/sklearnplus/_coverage.py tests/test_coverage_imports.py
git commit -m "chore: scaffold sklearnplus package"
```

---

### Task 2: sklearn supervised re-exports (linear_model → dummy)

**Files:**
- Create: `src/sklearnplus/linear_model/__init__.py`, `tree/__init__.py`, `neighbors/__init__.py`, `svm/__init__.py`, `naive_bayes/__init__.py`, `discriminant_analysis/__init__.py`, `neural_network/__init__.py`, `gaussian_process/__init__.py`, `dummy/__init__.py`, `kernel_ridge/__init__.py`, `ensemble/__init__.py` (sklearn-only first)
- Modify: `src/sklearnplus/_coverage.py`
- Test: `tests/test_reexport_identity.py`

**Interfaces:**
- Produces: each `__init__.py` defines `__all__` and re-exports listed names as **the same object** as sklearn.

**Symbol list (must all appear in `_coverage.PUBLIC_SYMBOLS`):**

- `linear_model`: LogisticRegression, LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor, SGDClassifier
- `tree`: DecisionTreeClassifier, DecisionTreeRegressor
- `neighbors`: KNeighborsClassifier, KNeighborsRegressor
- `svm`: SVC, SVR
- `naive_bayes`: GaussianNB
- `discriminant_analysis`: LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
- `neural_network`: MLPClassifier, MLPRegressor
- `gaussian_process`: GaussianProcessClassifier
- `dummy`: DummyClassifier, DummyRegressor
- `kernel_ridge`: KernelRidge
- `ensemble` (sklearn part): RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor, CalibratedClassifierCV

- [ ] **Step 1: Failing identity test**

```python
# tests/test_reexport_identity.py
from sklearn import linear_model as sk_lm
from sklearnplus import linear_model as sp_lm

def test_logistic_regression_is_sklearn():
    assert sp_lm.LogisticRegression is sk_lm.LogisticRegression
```

- [ ] **Step 2: Run — FAIL (module missing)**

Run: `pytest tests/test_reexport_identity.py::test_logistic_regression_is_sklearn -q`

- [ ] **Step 3: Implement re-export pattern** (repeat per submodule)

```python
# src/sklearnplus/linear_model/__init__.py
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    TheilSenRegressor,
)

__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "HuberRegressor",
    "Lars",
    "Lasso",
    "LassoLars",
    "LinearRegression",
    "LogisticRegression",
    "OrthogonalMatchingPursuit",
    "PassiveAggressiveRegressor",
    "RANSACRegressor",
    "Ridge",
    "RidgeClassifier",
    "SGDClassifier",
    "TheilSenRegressor",
]
```

Apply the same pattern to other submodules. Extend `_coverage.PUBLIC_SYMBOLS` with every `(module, name)` pair.

- [ ] **Step 4: Expand identity tests for one symbol per submodule; run coverage imports**

Run: `pytest tests/test_reexport_identity.py tests/test_coverage_imports.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sklearnplus tests
git commit -m "feat: re-export sklearn supervised estimators"
```

---

### Task 3: Boosting dual-path (xgboost / lightgbm / catboost)

**Files:**
- Create: `src/sklearnplus/xgboost/__init__.py`, `lightgbm/__init__.py`, `catboost/__init__.py`
- Modify: `src/sklearnplus/ensemble/__init__.py`, `_coverage.py`
- Test: `tests/test_boosting_dual_path.py`

**Interfaces:**
- Produces: `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `LGBMRegressor`, `CatBoostClassifier`, `CatBoostRegressor` from both `sklearnplus.ensemble` and dedicated modules; `sklearnplus.xgboost.XGBClassifier is sklearnplus.ensemble.XGBClassifier`.

- [ ] **Step 1: Failing dual-path test**

```python
# tests/test_boosting_dual_path.py
def test_xgb_dual_path_same_object():
    from sklearnplus.ensemble import XGBClassifier as A
    from sklearnplus.xgboost import XGBClassifier as B
    assert A is B

def test_lgbm_dual_path_same_object():
    from sklearnplus.ensemble import LGBMClassifier as A
    from sklearnplus.lightgbm import LGBMClassifier as B
    assert A is B

def test_catboost_dual_path_same_object():
    from sklearnplus.ensemble import CatBoostClassifier as A
    from sklearnplus.catboost import CatBoostClassifier as B
    assert A is B
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement**

```python
# src/sklearnplus/xgboost/__init__.py
from xgboost import XGBClassifier, XGBRegressor
__all__ = ["XGBClassifier", "XGBRegressor"]
```

```python
# src/sklearnplus/lightgbm/__init__.py
from lightgbm import LGBMClassifier, LGBMRegressor
__all__ = ["LGBMClassifier", "LGBMRegressor"]
```

```python
# src/sklearnplus/catboost/__init__.py
from catboost import CatBoostClassifier, CatBoostRegressor
__all__ = ["CatBoostClassifier", "CatBoostRegressor"]
```

In `ensemble/__init__.py`, also `from sklearnplus.xgboost import XGBClassifier, XGBRegressor` (and lightgbm/catboost) and add to `__all__`.

- [ ] **Step 4: pytest dual-path + coverage — PASS**

- [ ] **Step 5: Commit** `feat: dual-path boosting re-exports`

---

### Task 4: Cluster + kmodes

**Files:**
- Create: `src/sklearnplus/cluster/__init__.py`
- Modify: `_coverage.py`
- Test: extend `tests/test_reexport_identity.py`

**Symbols:** KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch, KModes

- [ ] **Step 1: Test `KModes` importable and `KMeans is sklearn.cluster.KMeans`**
- [ ] **Step 2: FAIL then implement re-exports** (`from kmodes.kmodes import KModes`)
- [ ] **Step 3: PASS + commit** `feat: re-export clustering estimators`

---

### Task 5: Anomaly adapters (pyod)

**Files:**
- Create: `src/sklearnplus/anomaly/_adapters.py`, `anomaly/__init__.py`
- Modify: `_coverage.py`
- Test: `tests/test_anomaly_adapter.py`

**Interfaces:**
- Produces: wrapper classes `ABOD`, `CBLOF`, `COF`, `IForest`, `HBOS`, `KNN`, `LOF`, `OCSVM`, `PCA`, `MCD`, `SOD`, `SOS` with `fit(X, y=None)`, `predict(X)`, `decision_function(X)` delegating to pyod; usable as final step in sklearn `Pipeline`.

- [ ] **Step 1: Failing pipeline smoke for IForest**

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearnplus.anomaly import IForest

def test_iforest_in_pipeline():
    X = np.random.randn(80, 4)
    pipe = Pipeline([("scaler", StandardScaler()), ("od", IForest(contamination=0.1))])
    pipe.fit(X)
    pred = pipe.predict(X)
    assert pred.shape == (80,)
```

- [ ] **Step 2: Implement thin adapter**

```python
# src/sklearnplus/anomaly/_adapters.py
from sklearn.base import BaseEstimator, OutlierMixin
import pyod.models.iforest as pyod_iforest
# ... one factory or per-model subclasses wrapping pyod detectors
class IForest(OutlierMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def fit(self, X, y=None):
        self.detector_ = pyod_iforest.IForest(**self.kwargs)
        self.detector_.fit(X)
        return self
    def predict(self, X):
        return self.detector_.predict(X)
    def decision_function(self, X):
        return self.detector_.decision_function(X)
    def get_params(self, deep=True):
        return dict(self.kwargs)
    def set_params(self, **params):
        self.kwargs.update(params)
        return self
```

Implement all 12 detectors similarly (shared `_PyODAdapter` base taking `detector_cls` to stay DRY). Export from `anomaly/__init__.py`.

- [ ] **Step 3: Tests PASS; commit** `feat: pyod anomaly adapters`

---

### Task 6: Sampling + pipeline helpers

**Files:**
- Create: `src/sklearnplus/sampling/__init__.py`, `pipeline/__init__.py`, `compose/__init__.py`, `impute/__init__.py`, `decomposition/__init__.py`, `feature_selection/__init__.py`, `model_selection/__init__.py`, `metrics/__init__.py`
- Modify: `_coverage.py`

**sampling `__all__`:** SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold, SMOTEENN, SMOTETomek

**pipeline:** re-export sklearn `Pipeline`, `make_pipeline`; also `ImbPipeline` from `imblearn.pipeline.Pipeline`

**compose:** ColumnTransformer, make_column_transformer, make_column_selector  
**impute:** SimpleImputer, KNNImputer, IterativeImputer  
**decomposition:** PCA, KernelPCA, IncrementalPCA  
**feature_selection:** VarianceThreshold, SelectKBest, SelectFromModel, SequentialFeatureSelector  
**model_selection:** GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score  
**metrics:** accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score, silhouette_score

- [ ] **Step 1: Tests that `SMOTE is imblearn...SMOTE` and `ImbPipeline` works with SMOTE + LogisticRegression**
- [ ] **Step 2: Implement re-exports; update coverage**
- [ ] **Step 3: PASS; commit** `feat: sampling and sklearn utility re-exports`

---

### Task 7: P0 custom preprocessors

**Files:**
- Create: `preprocessing/_clean_column_names.py`, `_date_features.py`, `_rare_category.py`, `_target_label.py`, `preprocessing/__init__.py` (also re-export StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, Normalizer, FunctionTransformer, and TargetEncoder if sklearn≥1.3)
- Test: `tests/test_preprocessing_p0.py`

**Interfaces:**
- `CleanColumnNames.fit/transform`: DataFrame → DataFrame, columns sanitized (`[^0-9a-zA-Z_]` → `_`)
- `DateFeatureExtractor(features=("year","month","day","dayofweek"))`: datetime columns → numeric feature columns; drop originals optional flag `drop_original=True`
- `RareCategoryGrouper(min_frequency=0.05, replacement="rare")`: for object/category cols
- `TargetLabelEncoder`: `fit(y)`, `transform(y)`, `inverse_transform(y)` for classification targets (wrap LabelEncoder; expose as estimator with `fit(X, y)` no-op on X if used carefully — prefer fit on y only documented; implement `fit(X, y)` storing encoder on `y` for pipeline-friendly use via custom API `fit(y)` **and** sklearn-style optional — spec: class with `fit(y)`, `transform(y)`, `inverse_transform`)

- [ ] **Step 1: Write failing tests for each P0 class (DataFrame roundtrip)**
- [ ] **Step 2: Implement until PASS; run `check_estimator` where applicable (CleanColumnNames, RareCategoryGrouper on DataFrame may need tags — document skips)**
- [ ] **Step 3: Commit** `feat: P0 preprocessing transformers`

---

### Task 8: End-to-end smoke + README

**Files:**
- Create: `tests/test_smoke_pipelines.py`
- Modify: `README.md`, ensure `_coverage.PUBLIC_SYMBOLS` complete vs Tasks 2–7

- [ ] **Step 1: Smoke tests**

```python
def test_clf_pipeline():
    from sklearn.datasets import load_breast_cancer
    from sklearnplus.pipeline import Pipeline
    from sklearnplus.preprocessing import StandardScaler
    from sklearnplus.ensemble import RandomForestClassifier
    X, y = load_breast_cancer(return_X_y=True)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=10, random_state=0))])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.9

def test_reg_pipeline():
    ...

def test_cluster_pipeline():
    ...

def test_anomaly_pipeline():
    ...
```

- [ ] **Step 2: README with install + import examples (no get_model)**
- [ ] **Step 3: Full pytest green; commit** `test: smoke pipelines and README`

---

### Task 9: Spec coverage gate (final)

- [ ] Diff `_coverage.PUBLIC_SYMBOLS` against design §5 lists; add any missing re-exports.
- [ ] Confirm no public `get_model` symbol exists (`pytest` negative test).

```python
def test_no_get_model():
    import sklearnplus
    assert not hasattr(sklearnplus, "get_model")
```

- [ ] Commit `test: lock coverage checklist and ban get_model`

---

## Out of scope for v0.1 (follow-up plan)

P1 transformers: GroupFeatures, RemoveMulticollinearity, RemoveOutliers, IterativeImputerPlus, TextEmbedder; Optuna/skopt wrappers; full check_estimator matrix polish; docs site.

---

## Self-review (plan vs spec)

| Spec item | Task |
|-----------|------|
| Umbrella re-exports, no get_model | 2–6, 9 |
| Boosting dual path | 3 |
| Cluster + KModes | 4 |
| Anomaly pyod | 5 |
| Sampling imblearn + ImbPipeline | 6 |
| P0 preprocessors | 7 |
| All-in-one deps | 1 `pyproject.toml` |
| CI coverage / smoke | 1, 8, 9 |
| No TS/NLP | omitted by design |
| P1 customs | deferred |

