import ast

import pytest

from healthrag.config import REPO_ROOT

RUNTIME_FILES = [
    REPO_ROOT / "app.py",
    *(
        REPO_ROOT / "healthrag" / f"{name}.py"
        for name in ("config", "retrieval", "logs", "llm", "conversation")
    ),
]
BUILD_MODULES = {"healthrag.corpus", "healthrag.build_index"}


def imported_modules(path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: p.name)
def test_runtime_code_never_imports_build_modules(path):
    assert not (imported_modules(path) & BUILD_MODULES)


@pytest.mark.parametrize("path", RUNTIME_FILES, ids=lambda p: p.name)
def test_only_app_imports_streamlit(path):
    uses_streamlit = any(m.split(".")[0] == "streamlit" for m in imported_modules(path))
    assert uses_streamlit == (path.name == "app.py")
