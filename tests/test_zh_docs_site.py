"""Content contract for the Chinese GitHub Pages landing and README links.

These checks are file-based so they run without installing sklplus.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"
NOJEKYLL = DOCS / ".nojekyll"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
PUBLISH = ROOT / ".github" / "workflows" / "publish-to-pypi.yml"
GITIGNORE = ROOT / ".gitignore"
README_EN = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
WIKI = ROOT / "wiki"

PAGES_URL = "https://incubatorshokuhou.github.io/sklplus/"
WIKI_URL = "https://github.com/IncubatorShokuhou/sklplus/wiki"
PYPI_URL = "https://pypi.org/project/sklplus/"
GITHUB_URL = "https://github.com/IncubatorShokuhou/sklplus"

FORBIDDEN_LANDING_PHRASES = (
    "是什么",
    "不适配",
    "get_model",
    "setup()",
    "不是 AutoML",
)


def _git_check_ignore(path: str) -> int:
    return subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT,
        check=False,
    ).returncode


def test_nojekyll_exists():
    assert NOJEKYLL.is_file()


def test_index_html_exists_and_is_chinese_landing():
    assert INDEX.is_file()
    html = INDEX.read_text(encoding="utf-8")
    assert 'lang="zh-CN"' in html
    assert "<h1" in html and "sklplus" in html
    assert "pip install sklplus" in html
    for symbol in ("Pipeline", "StandardScaler", "RandomForestClassifier"):
        assert symbol in html
    assert "pipe.fit" in html
    assert "pipe.score" in html
    for extra in (
        "CleanColumnNames",
        "RareCategoryGrouper",
        "SMOTE",
        "IForest",
        "ImbPipeline",
        "ColumnTransformer",
    ):
        assert extra not in html
    assert PYPI_URL in html
    assert GITHUB_URL in html
    assert WIKI_URL in html
    for phrase in FORBIDDEN_LANDING_PHRASES:
        assert phrase not in html
    assert "MkDocs" not in html
    assert "Sphinx" not in html


def test_pages_files_are_not_gitignored():
    for rel in ("docs/index.html", "docs/.nojekyll", "docs/styles.css"):
        assert _git_check_ignore(rel) == 1, f"{rel} should be tracked"


def test_superpowers_dir_stays_gitignored():
    assert "docs/superpowers/" in GITIGNORE.read_text(encoding="utf-8")
    assert _git_check_ignore("docs/superpowers/") == 0
    assert _git_check_ignore("docs/superpowers/specs/example.md") == 0


def test_pages_workflow_deploys_docs_via_official_actions():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/upload-pages-artifact" in text
    assert "actions/deploy-pages" in text
    assert "path: docs" in text or "path: 'docs'" in text or 'path: "docs"' in text
    assert "include-hidden-files: true" in text
    assert "pages: write" in text
    assert "id-token: write" in text
    assert "publish-to-pypi.yml" not in text


def test_pypi_trusted_publishing_workflow_untouched():
    text = PUBLISH.read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert "id-token: write" in text
    assert "environment:" in text
    assert "name: pypi" in text


def test_readmes_link_pages_and_wiki():
    for path in (README_EN, README_ZH):
        text = path.read_text(encoding="utf-8")
        assert PAGES_URL in text
        assert WIKI_URL in text


def test_wiki_source_pages_cover_required_topics():
    names = {
        "Home.md",
        "安装与环境.md",
        "导入约定.md",
        "预处理与采样.md",
        "示例索引.md",
        "_Sidebar.md",
    }
    present = {p.name for p in WIKI.glob("*.md")} | {p.name for p in WIKI.glob("_*.md")}
    assert names <= present
    assert "FAQ.md" not in present

    home = (WIKI / "Home.md").read_text(encoding="utf-8")
    assert PAGES_URL in home
    assert "get_model" not in home
    assert "本 Wiki" not in home
    assert "外部链接" not in home

    sidebar = (WIKI / "_Sidebar.md").read_text(encoding="utf-8")
    assert "FAQ" not in sidebar

    install = (WIKI / "安装与环境.md").read_text(encoding="utf-8")
    assert "pip install sklplus" in install
    assert "AutoML" not in install

    imports = (WIKI / "导入约定.md").read_text(encoding="utf-8")
    assert "ImbPipeline" in imports
    assert "from sklplus.xgboost import XGBClassifier as B" in imports
    assert "fit_resample" in imports

    prep = (WIKI / "预处理与采样.md").read_text(encoding="utf-8")
    assert "## P0" not in prep
    assert "## P1" not in prep
    for heading in (
        "列名",
        "稀有类别",
        "日期",
        "标签",
        "分组统计",
        "相关",
        "插补",
        "文本列",
        "异常行",
    ):
        assert heading in prep
    for name in (
        "CleanColumnNames",
        "RareCategoryGrouper",
        "DateFeatureExtractor",
        "TargetLabelEncoder",
        "GroupFeatures",
        "RemoveMulticollinearity",
        "IterativeImputerPlus",
        "TextEmbedder",
        "RemoveOutliers",
    ):
        assert name in prep

    examples = (WIKI / "示例索引.md").read_text(encoding="utf-8")
    for i in range(1, 7):
        assert f"0{i}_" in examples
    assert "最小可跑示例" in examples
    assert "P0" not in examples
    assert "P1" not in examples
