# sklearnplus

[English README](README.md)

面向表格机器学习的 **大号 sklearn**：按 sklearn 风格路径再导出常用估计器，并在生态缺口处提供薄适配与少量自写预处理器。

这是 **组件库**，不是 AutoML。**没有** `get_model`、`setup` 或按字符串 ID 取模型的工厂——直接 import 类，自己拼 `Pipeline`。

默认安装即包含 boosting（xgboost / lightgbm / catboost）、imbalanced-learn、pyod、kmodes、category-encoders、feature-engine，清单内 import 开箱可用。

## 安装

```bash
pip install -e ".[dev]"
```

## 设计理念

| 做 | 不做 |
|----|------|
| `from sklearnplus.ensemble import RandomForestClassifier` | `get_model("rf")` |
| 组合 `Pipeline` / `ImbPipeline` | 全局 `setup()` 会话状态 |
| 再导出时与上游是同一对象 | 改变算法语义的厚包装 |

## Boosting 双路径

XGBoost / LightGBM / CatBoost 可从 `sklearnplus.ensemble` 与独立子模块导入——**同一类对象**：

```python
from sklearnplus.ensemble import XGBClassifier as A
from sklearnplus.xgboost import XGBClassifier as B
assert A is B
```

## 示例 import

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

## 可运行示例

见 [`examples/`](examples/)：

| 脚本 | 内容 |
|------|------|
| `01_classification_pipeline.py` | `Pipeline` + 标准化 + 随机森林 |
| `02_imbalanced_imbpipeline.py` | `ImbPipeline` + SMOTE + 逻辑回归 |
| `03_boosting_dual_path.py` | 双路径同一性 + 小规模 XGB 拟合 |
| `04_preprocessing_p0.py` | P0 自写变换器（DataFrame） |
| `05_anomaly_iforest.py` | 标准化 + `IForest` 异常检测 |

```bash
python examples/01_classification_pipeline.py
```

## 范围（v0.1）

- 分类、回归、聚类、异常检测
- 预处理 / 采样 / 特征选择 / 搜索 / 指标辅助
- P0 自写：`CleanColumnNames`、`DateFeatureExtractor`、`RareCategoryGrouper`、`TargetLabelEncoder`

**不在 v0.1：** 时序、NLP 专题模块。

## 延后（P1）

`GroupFeatures`、`RemoveMulticollinearity`、`RemoveOutliers`、
`IterativeImputerPlus`、`TextEmbedder`；Optuna/skopt 封装；更完整的
`check_estimator`；文档站点。
