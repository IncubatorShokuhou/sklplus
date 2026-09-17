# sklearnplus

**Big sklearn** for tabular ML: re-export familiar estimators under sklearn-style
paths, plus a few thin adapters and P0 preprocessors where the ecosystem has gaps.

This is a **component library**, not AutoML. There is **no** `get_model`,
`setup`, or string-ID model factory — you import classes and compose
`Pipeline`s yourself.

Default `pip install` pulls boosting (xgboost / lightgbm / catboost),
imbalanced-learn, pyod, kmodes, category-encoders, and feature-engine so listed
imports work out of the box.

## Install

```bash
pip install -e ".[dev]"
```

## Example imports

```python
from sklearnplus.linear_model import LogisticRegression, Ridge
from sklearnplus.ensemble import (
    RandomForestClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
)
# Boosting dual-path: same class object as ensemble
from sklearnplus.xgboost import XGBClassifier as XGBClassifier2
assert XGBClassifier is XGBClassifier2

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

# Resampling-aware pipeline
pipe = ImbPipeline([
    ("sample", SMOTE(random_state=0)),
    ("clf", LogisticRegression(max_iter=500)),
])
```

## Scope (v0.1)

- Classification, regression, clustering, anomaly detection
- Preprocess / sample / select / search / metrics helpers
- **Not** time series or NLP specialist modules
