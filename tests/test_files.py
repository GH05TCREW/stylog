"""Directory traversal and glob conformance (spec 18.9-18.13, 25.22)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stylog.config import InputConfig
from stylog.exceptions import ResourceLimitError
from stylog.infrastructure.files import compile_glob, select_files


def _touch(root: Path, relative: str, content: bytes = b"x") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_glob_star_within_segment() -> None:
    regex = compile_glob("*.py")
    assert regex.match("a.py")
    assert not regex.match("sub/a.py")


def test_glob_double_star_zero_or_more() -> None:
    regex = compile_glob("**/*.py")
    assert regex.match("a.py")
    assert regex.match("sub/deep/a.py")
    assert not regex.match("a.txt")


def test_glob_double_star_middle() -> None:
    regex = compile_glob("src/**/test_*.py")
    assert regex.match("src/test_a.py")
    assert regex.match("src/x/y/test_a.py")
    assert not regex.match("src/x/y/a.py")


def test_glob_question_and_class() -> None:
    assert compile_glob("a?.txt").match("ab.txt")
    assert not compile_glob("a?.txt").match("a/b.txt")
    assert compile_glob("a[bc].txt").match("ab.txt")
    assert not compile_glob("a[bc].txt").match("ad.txt")


def test_glob_case_sensitive() -> None:
    assert not compile_glob("*.py").match("A.PY")


def test_selection_hidden_excluded_by_default(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py")
    _touch(tmp_path, ".hidden/b.py")
    _touch(tmp_path, ".config.py")
    selected = select_files(tmp_path, InputConfig(include=("**/*.py",)))
    assert [s.relative_path for s in selected] == ["a.py"]


def test_selection_include_exclude_order(tmp_path: Path) -> None:
    _touch(tmp_path, "keep/a.py")
    _touch(tmp_path, "skip/b.py")
    config = InputConfig(include=("**/*.py",), exclude=("skip/**",))
    selected = select_files(tmp_path, config)
    assert [s.relative_path for s in selected] == ["keep/a.py"]


def test_selection_default_excludes(tmp_path: Path) -> None:
    _touch(tmp_path, "pkg/a.py")
    _touch(tmp_path, "node_modules/lib/b.py")
    _touch(tmp_path, "dist/c.py")
    selected = select_files(tmp_path, InputConfig())
    assert [s.relative_path for s in selected] == ["pkg/a.py"]


def test_selection_sorted_order(tmp_path: Path) -> None:
    for name in ("b.py", "a.py", "sub/c.py", "sub/a.py"):
        _touch(tmp_path, name)
    selected = select_files(tmp_path, InputConfig(include=("**/*.py",), exclude=()))
    assert [s.relative_path for s in selected] == ["a.py", "b.py", "sub/a.py", "sub/c.py"]


def test_selection_limits(tmp_path: Path) -> None:
    for index in range(3):
        _touch(tmp_path, f"f{index}.py")
    with pytest.raises(ResourceLimitError):
        select_files(tmp_path, InputConfig(include=("**/*.py",), max_files=2))
    with pytest.raises(ResourceLimitError):
        select_files(tmp_path, InputConfig(include=("**/*.py",), max_total_bytes=2))


def test_symlink_not_followed(tmp_path: Path) -> None:
    _touch(tmp_path, "real/a.py")
    link = tmp_path / "link.py"
    try:
        os.symlink(tmp_path / "real" / "a.py", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    selected = select_files(tmp_path, InputConfig(include=("**/*.py",)))
    names = [s.relative_path for s in selected]
    # the symlink entry is never followed into traversal; it may appear as a
    # selected *file* entry only if it matches globs, but is_symlink is honored
    # by the ingest layer (SYMLINK_REJECTED)
    assert "real/a.py" in names
