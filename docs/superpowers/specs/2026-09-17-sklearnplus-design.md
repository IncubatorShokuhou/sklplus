# sklearnplus 设计文档

**日期：** 2026-09-17  
**状态：** 已批准（2026-09-17）  
**包名：** `sklearnplus`  
**一句话：** 大号 sklearn——把表格 ML 常用库按 sklearn 风格路径再导出/轻包装，缺什么补什么，不做 AutoML。

---

## 1. 背景与目标

用户认为 PyCaret 太重，像开箱软件，而不是 sklearn 训练流水线组件。需要的是：

- 能 `from sklearnplus.xxx import Yyy` 的 **transformer / estimator**
- 完全贴合 sklearn API（`fit` / `transform` / `predict`、`get_params` / `set_params`、可 `clone`、可进 `Pipeline`）
- 覆盖面参考 PyCaret 的表格能力，但 **API 形态是 sklearn，不是 Experiment / setup**

**成功标准**

1. 覆盖清单上的每个公开类都能直接 import，且能塞进 `sklearn.pipeline.Pipeline`（有采样时用 `imblearn.pipeline.Pipeline`）。
2. 默认 `pip install sklearnplus` 后，清单内 import 不因缺依赖而失败（一把梭依赖）。
3. 自定义类通过（或明确记录例外的）`check_estimator`；再导出的第三方类以上游行为为准。
4. **不做** 时序、NLP 专题模块、`setup` 全局状态、`get_model(id)`、看板 / plot / deploy。

---

## 2. 非目标

| 不做 | 原因 |
|------|------|
| `get_model("lr", task=...)` | 用户明确不要字符串 ID 工厂 |
| PyCaret `setup` / `compare_models` / `pull` | 会话式 AutoML，不是组件库 |
| `plot_model` / dashboard / deploy / create_app | UI / MLOps，不是 estimator |
| 时序（sktime 等）、NLP 专题 | 用户明确排除 |
| 复刻 PyCaret 控制面（FastAPI / React） | 与「轻包装」相反 |

---

## 3. 方案选择

采用 **方案 A：再导出 + 薄封装 + 缺口自写**。

- sklearn / xgboost / lightgbm / catboost / imblearn / pyod / kmodes / category-encoders / feature-engine 等：**优先再导出**（同一对象或极薄子类，不改算法语义）。
- 仅当 PyCaret 3.x 预处理有而生态没有干净等价物时：**自写** `BaseEstimator` + `TransformerMixin`。

不采用：从 PyCaret 拆 Experiment 容器（方案 B）；不以 skrub 替换全部预处理语义（方案 C，可作可选加速路径，非默认）。

---

## 4. 包结构与 import 约定

### 4.1 顶层布局

```
sklearnplus/
  __init__.py                 # 版本号；不污染命名空间
  linear_model/
  tree/
  neighbors/
  svm/
  naive_bayes/
  discriminant_analysis/
  neural_network/
  ensemble/                   # sklearn 集成 + 同时再导出 boosting
  dummy/
  cluster/
  anomaly/                    # pyod 等异常检测（轻适配）
  decomposition/
  preprocessing/              # sklearn 再导出 + 自写缺口
  impute/
  feature_selection/
  compose/
  pipeline/
  model_selection/
  metrics/
  sampling/                   # imblearn 再导出
  xgboost/                    # 独立子模块（与 ensemble 双路径）
  lightgbm/
  catboost/
  _coverage.py                # 仅测试/CI：覆盖清单，不作为用户 API
```

### 4.2 Import 风格（唯一用户 API）

```python
from sklearnplus.linear_model import LogisticRegression, Ridge
from sklearnplus.ensemble import (
    RandomForestClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
)
from sklearnplus.xgboost import XGBClassifier as XGBClassifier2  # 等价
from sklearnplus.preprocessing import StandardScaler, RareCategoryGrouper
from sklearnplus.sampling import SMOTE
from sklearnplus.anomaly import IForest
from sklearnplus.pipeline import Pipeline
from sklearnplus.compose import ColumnTransformer
```

**双路径规则（boosting）**

- `sklearnplus.ensemble.XGBClassifier` ≡ `sklearnplus.xgboost.XGBClassifier`（同一对象或文档保证行为一致）。
- LightGBM、CatBoost 同理。

**禁止**

- `get_model` / `create_model` / 按 PyCaret 短 ID 取类的公共 API。

### 4.3 再导出实现约定

- 优先：`from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier` 再在 `__all__` 列出。
- 需要统一 DataFrame 列名 / 标签时，才用极薄子类；默认不改上游类。
- 异常检测（pyod）：提供薄适配，使 `fit(X, y=None)` / `predict(X)` 在 Pipeline 中可用；文档写明与纯 sklearn 的差异（如 `decision_function`）。

---

## 5. 覆盖范围（对齐 PyCaret 表格能力，排除时序/NLP）

### 5.1 分类估计器（再导出）

LogisticRegression, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, SGDClassifier（线性 SVM 路径）, SVC, GaussianProcessClassifier, MLPClassifier, RidgeClassifier, RandomForestClassifier, QuadraticDiscriminantAnalysis, AdaBoostClassifier, GradientBoostingClassifier, LinearDiscriminantAnalysis, ExtraTreesClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier, DummyClassifier；以及 BaggingClassifier, StackingClassifier, VotingClassifier, CalibratedClassifierCV。

### 5.2 回归估计器（再导出）

LinearRegression, Lasso, Ridge, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor, KernelRidge, SVR, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, MLPRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor, DummyRegressor；以及 Bagging/Stacking/Voting Regressor。

### 5.3 聚类（再导出）

KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch, KModes（kmodes）。

### 5.4 异常检测（pyod 轻适配）

ABOD, CBLOF, COF, IForest, HBOS, KNN, LOF, OCSVM, PCA, MCD, SOD, SOS（类名与文档表对齐 PyCaret 能力，import 路径在 `sklearnplus.anomaly`）。

### 5.5 预处理：再导出

SimpleImputer, KNNImputer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder（sklearn≥1.3）, PolynomialFeatures, VarianceThreshold, PCA, KernelPCA, IncrementalPCA, KBinsDiscretizer, CountVectorizer / TfidfVectorizer（经 ColumnTransformer 使用的辅助）, LabelEncoder（仅文档说明慎用于 y）, SelectKBest, SelectFromModel, SequentialFeatureSelector 等与 PyCaret 步骤对应的 sklearn 符号。

### 5.6 预处理：自写缺口（P0/P1）

| 类名 | 职责 | 优先级 |
|------|------|--------|
| `CleanColumnNames` | 清洗列名 | P0 |
| `DateFeatureExtractor` | 日期列展开 | P0 |
| `RareCategoryGrouper` | 低频类别合并 | P0 |
| `GroupFeatures` | 分组聚合特征 | P1 |
| `RemoveMulticollinearity` | 相关过滤 | P1 |
| `RemoveOutliers` | 仅训练集去异常行 | P1 |
| `IterativeImputerPlus` | 数值+类别迭代插补 | P1 |
| `TextEmbedder` | 按列 bow/tfidf 并拼回 | P1 |
| `TargetLabelEncoder` | 分类 y 编解码（配合预测反变换） | P0 |

实现基类：`sklearn.base.BaseEstimator` + `TransformerMixin`（或适当 Mixin）；支持 `get_params` / `set_params`；尽量输出 pandas DataFrame（若输入是 DataFrame），与 sklearn 1.x `set_output` 策略兼容处优先跟 sklearn。

### 5.7 采样（imblearn 再导出）

SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold, SMOTEENN, SMOTETomek；以及 `sklearnplus.pipeline` 中再导出 `imblearn.pipeline.Pipeline`（名称为 `ImbPipeline` 或文档明确 `from sklearnplus.sampling.pipeline import Pipeline`——实现时二选一，避免与 sklearn Pipeline 混淆：**推荐** `from sklearnplus.pipeline import Pipeline` 为 sklearn，`from sklearnplus.pipeline import make_imb_pipeline` / `ImbPipeline` 为 imblearn）。

### 5.8 模型选择与指标

- 再导出：`GridSearchCV`, `RandomizedSearchCV`, 常用 `make_scorer`。
- 可选再导出（同属一把梭依赖若已列入）：Optuna / skopt 搜索封装可放 `model_selection` 后期；首版以 sklearn 搜索为主。
- `metrics`：再导出常用分类/回归/聚类指标；维护与 PyCaret 常见名称对应的 **文档表**（不是字符串 `optimize="acc"` API）。

---

## 6. 依赖策略

**一把梭（用户已选）：** `pip install sklearnplus` 默认安装覆盖清单所需依赖，包括但不限于：

- scikit-learn, numpy, pandas, scipy, joblib
- xgboost, lightgbm, catboost
- imbalanced-learn
- pyod, numba（若 pyod 需要）
- kmodes
- category-encoders, feature-engine（自写可依赖或逐步减少）

可选后续：`[dev]` 含 pytest、ruff 等。不在首版用 extras 拆 boosting（与「import 都能用」一致）。

---

## 7. 验收与「不能漏」

1. **`_coverage.py` / 测试清单**：枚举所有应对外公开的符号；CI 断言每个符号可 import。
2. **对照表测试**：维护「PyCaret 表格能力 → sklearnplus 符号」映射；缺映射则失败。（映射仅测试用，不是用户 `get_model`。）
3. **Pipeline 冒烟**：分类 / 回归 / 聚类 / 异常各至少一条端到端 Pipeline。
4. **`check_estimator`**：所有自写类；失败必须在文档「已知限制」中登记。
5. **不做** 与 PyCaret 数值结果逐位一致（允许实现差异）；要求的是 **能力可及 + API 可组合**。

---

## 8. 交付与仓库

- 新建项目（Cursor Origin / 新仓库），Python 包 `sklearnplus`。
- 首版里程碑：脚手架 + 再导出全量模型符号 + P0 自写预处理 + CI 覆盖断言 + 冒烟测试。
- 次版：P1 自写预处理、异常适配完善、文档站点。

---

## 9. 开放问题（实现前可默认）

| 问题 | 默认 |
|------|------|
| 命名空间是否镜像 sklearn 全部子模块 | 首版只建覆盖清单需要的子模块，不求 100% 镜像 sklearn 全集 |
| DataFrame 出入 | 输入 DataFrame 则尽量保持列；否则 numpy |
| pyod 类名风格 | 短名 `IForest` 与全名并存或只暴露常用别名——实现时以清晰 `__all__` 为准 |

---

## 10. 修订记录

- 2026-09-17：初稿。方案 A；包名 sklearnplus；排除 get_model；boosting 双路径 import；依赖一把梭；范围不含时序/NLP。
