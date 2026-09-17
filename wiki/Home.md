# sklplus

sklplus 0.1.1 是一层薄的表格估计器聚合包。多数符号是再导出（常常与上游是同一对象）；少量适配器和自写预处理器用来补我们手拼 `Pipeline` 时缺的那几块。

面向需要 `from sklplus.… import …`、按标准 `fit` / `transform` / `predict` 组合流水线的人。不是 AutoML：没有 `get_model`、没有 `setup()`、也没有按字符串 ID 取模型的工厂。

## 本 Wiki

- [[安装与环境]]
- [[导入约定]]
- [[预处理与采样]]
- [[示例索引]]
- [[FAQ]]

以后模块说明往这里追加。这里不堆完整 API 参考。

## 外部链接

- 中文项目页：https://incubatorshokuhou.github.io/sklplus/
- 源码：https://github.com/IncubatorShokuhou/sklplus
- PyPI：https://pypi.org/project/sklplus/
- 英文 README：https://github.com/IncubatorShokuhou/sklplus/blob/master/README.md
- 中文 README：https://github.com/IncubatorShokuhou/sklplus/blob/master/README.zh-CN.md
