# 贡献指南

感谢关注 sklplus。本仓库欢迎 issue 与 PR；请先读完本节再动手改代码。

## 本地开发

建议 Python `>=3.10`。在仓库根目录可编辑安装（含测试工具）：

```bash
pip install -e ".[dev]"
```

`[dev]` 会额外安装 `pytest>=7` 和 `ruff`。

## 测试

改动后请在本地跑通测试：

```bash
pytest
```

默认测试路径见 `pyproject.toml` 的 `[tool.pytest.ini_options]`。

## 请勿提交的路径

- **`docs/superpowers/`**：本地设计/计划笔记，已在 `.gitignore` 中排除，**永远不要**提交或推送。
- GitHub Pages 工作流会在部署前删除该目录（若误入工作区）。

## Wiki

仓库根目录的 **`wiki/`** 是 [GitHub Wiki](https://github.com/IncubatorShokuhou/sklplus/wiki) 的本地镜像，用于对照与同步，**不是**独立文档源。Wiki 内容请在 GitHub Wiki 侧维护，再按需同步到 `wiki/`。

## PR 建议

- 从最新 `master` 开分支。
- 保持改动聚焦；不要顺手发明新的 Pipeline 冲突规则，除非 issue 明确要求。
- CI（`.github/workflows/tests.yml`）会对 PR 跑 pytest（Python 3.10 / 3.12）。
