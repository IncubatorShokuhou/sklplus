# sklplus

[English README](README.md) · [中文项目页](https://incubatorshokuhou.github.io/sklplus/) · [Wiki](https://github.com/IncubatorShokuhou/sklplus/wiki)

sklplus 0.1.1 是一层薄的表格估计器聚合包。多数符号是再导出（常常与上游是同一对象）；少量适配器和自写预处理器用来补我们手拼 `Pipeline` 时缺的那几块。

面向需要 `from sklplus.… import …`、按标准 `fit` / `transform` / `predict` 组合流水线的人。不是 AutoML：没有 `get_model`、没有 `setup()`、也没有按字符串 ID 取模型的工厂。

**适用：** 表格上的分类、回归、聚类、异常检测，以及预处理 / 采样 / 特征选择 / 搜索 / 指标等按 sklearn 风格路径再导出的辅助符号。

**不适用（v0.1）：** 时序、NLP 专题模块。若你要的是 PyCaret 那种实验会话 API，请用别的工具。

## 依赖

- Python `>=3.10`
- `scikit-learn>=1.3`
- 默认安装还会带上：`numpy`、`pandas`、`scipy`、`joblib`、`xgboost`、`lightgbm`、`catboost`、`imbalanced-learn`、`pyod`、`kmodes`、`category-encoders`、`feature-engine`

## 安装

PyPI / import / GitHub 仓库名均为 **`sklplus`**（https://github.com/IncubatorShokuhou/sklplus）。

```bash
pip install sklplus
```

从克隆目录可编辑安装（含测试工具）：

```bash
pip install -e ".[dev]"
```

`[dev]` 额外安装 `pytest>=7` 和 `ruff`。

## 最小用法

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

再导出在能做到时与上游保持同一对象，例如 `sklplus.ensemble.RandomForestClassifier is sklearn.ensemble.RandomForestClassifier`。

### Boosting 的两条 import 路径

`XGBClassifier` / `LGBMClassifier` / `CatBoostClassifier`（以及对应的 Regressor）同时出现在 `sklplus.ensemble` 与 `sklplus.xgboost` / `lightgbm` / `catboost`。它们是同一个类对象：

```python
from sklplus.ensemble import XGBClassifier as A
from sklplus.xgboost import XGBClassifier as B
assert A is B
```

### 重采样

步骤里有 `fit_resample`（例如 `SMOTE`）时用 `ImbPipeline`。普通 `Pipeline` 仍是 sklearn 那份。

## 示例

[`examples/`](examples/) 下的脚本：

| 脚本 | 内容 |
|------|------|
| `01_classification_pipeline.py` | `Pipeline` + 标准化 + 随机森林 |
| `02_imbalanced_imbpipeline.py` | `ImbPipeline` + SMOTE + 逻辑回归 |
| `03_boosting_dual_path.py` | 双路径同一性 + 小规模 XGB 拟合 |
| `04_preprocessing_p0.py` | P0 自写变换器（DataFrame） |
| `05_anomaly_iforest.py` | 标准化 + `IForest` |

```bash
python examples/01_classification_pipeline.py
```

## 0.1.1 里的自写预处理

在 `sklplus.preprocessing`（与再导出的 scaler/encoder 一起）：

**P0**

- `CleanColumnNames`
- `DateFeatureExtractor`
- `RareCategoryGrouper`
- `TargetLabelEncoder`

**P1**

- `GroupFeatures` — 按列组做行内聚合统计
- `RemoveMulticollinearity` — 去掉高相关数值特征（可选按与 `y` 的相关保留）
- `IterativeImputerPlus` — 数值用 `IterativeImputer`，类别用众数
- `TextEmbedder` — 文本列 BoW / TF-IDF 后拼回

**采样（不在 preprocessing）：** `sklplus.sampling.RemoveOutliers` 是 imblearn 风格采样器（`fit_resample`），只在训练集去异常行 —— 请配合 `ImbPipeline`。

`sklplus.anomaly` 下是对 pyod 的薄包装（`IForest`、`LOF` 等），以便放进 sklearn `Pipeline`。

## 已知缺口

仍缺：Optuna/skopt 搜索封装；更完整的 `check_estimator`。P1 自写预处理 / `RemoveOutliers` 采样器已实现。

许可证：MIT（见 `LICENSE`）。

贡献方式：TODO。
