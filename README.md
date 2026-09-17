# sklearnplus

Big sklearn: re-exports and light wrappers for tabular ML pipelines.
Import estimators under familiar sklearn-style paths; default install includes
boosting, imblearn, pyod, kmodes, and related deps.

## Install

```bash
pip install -e ".[dev]"
```

## Example imports

```python
from sklearnplus.linear_model import LogisticRegression, Ridge
from sklearnplus.ensemble import RandomForestClassifier, XGBClassifier
from sklearnplus.xgboost import XGBClassifier  # dual-path with ensemble
from sklearnplus.preprocessing import StandardScaler
from sklearnplus.sampling import SMOTE
from sklearnplus.anomaly import IForest
from sklearnplus.pipeline import Pipeline
```
