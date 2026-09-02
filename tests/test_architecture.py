"""Architecture boundary enforcement (spec 4.15, 26.10). Release gate."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "stylog"

HEAVY_OR_FORBIDDEN_DOMAIN = {
    "typer",
    "tree_sitter",
    "spacy",
    "sklearn",
    "scipy",
    "pyarrow",
    "polars",
    "duckdb",
    "pandas",
    "numpy",
    "torch",
    "transformers",
}

ANALYSIS_FORBIDDEN = HEAVY_OR_FORBIDDEN_DOMAIN | {"typer"}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:  # module level only; function-level lazy imports allowed
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _modules_under(subdir: str) -> list[Path]:
    return sorted((SRC / subdir).rglob("*.py"))


def test_domain_imports_nothing_forbidden() -> None:
    for path in _modules_under("domain"):
        names = _top_level_imports(path)
        assert not (names & HEAVY_OR_FORBIDDEN_DOMAIN), f"{path}: {names & HEAVY_OR_FORBIDDEN_DOMAIN}"
        for forbidden in ("stylog.infrastructure", "stylog.cli", "stylog.application"):
            tree_text = path.read_text(encoding="utf-8")
            assert forbidden not in tree_text, f"{path} references {forbidden}"


def test_analysis_top_level_imports_stay_scientific() -> None:
    for path in _modules_under("analysis"):
        if path.name == "linguistic.py":
            # spaCy must be function-level lazy; verify no top-level spacy import.
            names = _top_level_imports(path)
            assert "spacy" not in names
            continue
        names = _top_level_imports(path)
        assert not (names & ANALYSIS_FORBIDDEN), f"{path}: {names & ANALYSIS_FORBIDDEN}"
        text = path.read_text(encoding="utf-8")
        assert "stylog.cli" not in text


def test_verification_top_level_imports_stay_stdlib_scientific() -> None:
    # The fitting stack must be pure-Python deterministic: no NumPy/BLAS, no
    # sklearn, no optional heavy modules at top level (spec 23.10-23.11).
    for path in _modules_under("verification"):
        names = _top_level_imports(path)
        assert not (names & ANALYSIS_FORBIDDEN), f"{path}: {names & ANALYSIS_FORBIDDEN}"
        text = path.read_text(encoding="utf-8")
        assert "stylog.cli" not in text


def test_verification_modules_load_no_optional_modules() -> None:
    code = (
        "import stylog.analysis.verify, stylog.verification.spec, "
        "stylog.verification.fit, stylog.application.verify, sys\n"
        "heavy = ['spacy','sklearn','scipy','pyarrow','polars','duckdb','pandas',"
        "'typer','torch','transformers']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )
    assert result.returncode == 0, result.stderr


def test_base_import_loads_no_optional_or_cli_modules() -> None:
    code = (
        "import stylog, sys\n"
        "heavy = ['spacy','sklearn','scipy','pyarrow','polars','duckdb','pandas',"
        "'typer','tree_sitter','torch','transformers']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )
    assert result.returncode == 0, result.stderr


def test_api_import_loads_no_optional_modules() -> None:
    code = (
        "import stylog.api, sys\n"
        "heavy = ['spacy','sklearn','pyarrow','polars','duckdb','pandas','typer']\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )
    assert result.returncode == 0, result.stderr


def test_no_plugin_or_enterprise_abstractions_exist() -> None:
    forbidden_names = (
        "NumericBackend",
        "CorpusStore",
        "AnalyzerFactory",
        "BackendManager",
        "PluginManager",
        "ProviderRegistry",
        "ServiceLocator",
        "PipelineBuilder",
        "UnitOfWork",
        "MessageBus",
    )
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden_names:
            assert f"class {name}" not in text, f"{path} defines {name}"
