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
