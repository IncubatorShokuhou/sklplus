# FAQ

## 这是 AutoML 吗？

不是。没有 `get_model`、没有 `setup()`、也没有按字符串 ID 取模型的工厂。用法就是 `from sklplus.… import …`，再按 `fit` / `transform` / `predict` 自己拼流水线。

## 和 sklearn 是什么关系？

薄的表格估计器聚合包。多数符号是再导出，能做到时与上游是同一对象（例如 `sklplus.ensemble.RandomForestClassifier is sklearn.ensemble.RandomForestClassifier`）。多出来的是少量适配器（如 `sklplus.anomaly` 对 pyod 的薄包装）和自写预处理 / 采样器，用来补手拼 `Pipeline` 时缺的那几块。

## 和 PyCaret 呢？

若你要的是 PyCaret 那种实验会话 API，请用别的工具。sklplus 不提供会话、不按字符串拉模型。

## v0.1 有时序或 NLP 专题模块吗？

没有。不适用时序、NLP 专题模块。`TextEmbedder` 只是把表格里的文本列做 BoW / TF-IDF 再拼回去，不是 NLP 专题栈。

## 包名 / 仓库名 / import 名是什么？

都是 **sklplus**。安装：`pip install sklplus`。PyPI：https://pypi.org/project/sklplus/

## 什么时候用 `ImbPipeline`？

步骤里有 `fit_resample` 时用，例如 `SMOTE` 或 `RemoveOutliers`。普通 `Pipeline` 仍是 sklearn 那份。

## Boosting 为什么有两条 import 路径？

`XGBClassifier` / `LGBMClassifier` / `CatBoostClassifier`（以及对应 Regressor）同时出现在 `sklplus.ensemble` 与各自的 `sklplus.xgboost` / `lightgbm` / `catboost`。它们是同一个类对象，不是两套实现。见 [[导入约定]]。

## 0.1.1 还缺什么？

仍缺：Optuna/skopt 搜索封装；更完整的 `check_estimator`。P1 自写预处理 / `RemoveOutliers` 采样器已实现。
