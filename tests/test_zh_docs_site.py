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

README_IMPORTS = [
    "from sklplus.linear_model import LogisticRegression, Ridge",
    "from sklplus.ensemble import RandomForestClassifier, XGBClassifier",
    "from sklplus.preprocessing import (",
    "    StandardScaler,",
    "    CleanColumnNames,",
    "    RareCategoryGrouper,",
    "    DateFeatureExtractor,",
    "    TargetLabelEncoder,",
    "from sklplus.sampling import SMOTE",
    "from sklplus.anomaly import IForest",
    "from sklplus.pipeline import Pipeline, ImbPipeline",
    "from sklplus.compose import ColumnTransformer",
]


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
    assert "0.1.1" in html
    assert "pip install sklplus" in html
    for heading in ("是什么", "不适配", "安装", "最短用法"):
        assert heading in html
    for snippet in README_IMPORTS:
        assert snippet in html
    assert PYPI_URL in html
    assert GITHUB_URL in html
    assert WIKI_URL in html
    assert "get_model" in html and "setup" in html
    assert "不是 AutoML" in html or "不是 AutoML" in html.replace("：", ":")
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
        "FAQ.md",
        "_Sidebar.md",
    }
    present = {p.name for p in WIKI.glob("*.md")} | {p.name for p in WIKI.glob("_*.md")}
    assert names <= present

    home = (WIKI / "Home.md").read_text(encoding="utf-8")
    assert "不是 AutoML" in home
    assert "get_model" in home
    assert PAGES_URL in home

    imports = (WIKI / "导入约定.md").read_text(encoding="utf-8")
    assert "ImbPipeline" in imports
    assert "from sklplus.xgboost import XGBClassifier as B" in imports
    assert "fit_resample" in imports

    prep = (WIKI / "预处理与采样.md").read_text(encoding="utf-8")
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
    for i in range(1, 6):
        assert f"0{i}_" in examples

    faq = (WIKI / "FAQ.md").read_text(encoding="utf-8")
    assert "不是 AutoML" in faq or "这是 AutoML" in faq
    assert "PyCaret" in faq
    assert "时序" in faq
    assert "NLP" in faq
