"""Deterministic directory traversal and glob selection (spec 18.9-18.13).

Glob semantics: case-sensitive on all OSes, ``/`` separators, ``*``/``?``/
``[abc]`` within one segment, ``**`` as a whole segment matching zero or more
segments. Patterns are root-anchored. Hidden components (leading dot) are
excluded before glob matching; exclude wins over include.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from stylog.config import InputConfig
from stylog.exceptions import ResourceLimitError


def _segment_regex(segment: str) -> str:
    """One glob segment: ``*``/``?``/``[abc]`` never cross ``/``."""
    out: list[str] = []
    index = 0
    n = len(segment)
    while index < n:
        char = segment[index]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = index + 1
            if end < n and segment[end] == "!":
                end += 1
            if end < n and segment[end] == "]":
                end += 1
            while end < n and segment[end] != "]":
                end += 1
            if end >= n:
                out.append(re.escape("["))
            else:
                inner = segment[index + 1 : end]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append("[" + inner.replace("\\", "\\\\") + "]")
                index = end
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a portable glob to a regex over normalized POSIX relative paths."""
    segments = pattern.split("/")
    parts: list[str] = []
    for index, segment in enumerate(segments):
        if segment == "**":
            if index == len(segments) - 1:
                parts.append("(?:[^/]+/)*[^/]*")
            else:
                parts.append("(?:[^/]+/)*")
            continue
        rendered = _segment_regex(segment)
        if index < len(segments) - 1:
            parts.append(rendered + "/")
        else:
            parts.append(rendered)
    return re.compile("^" + "".join(parts) + "$")


def _build_matcher(patterns: tuple[str, ...]):
    compiled = [compile_glob(pattern) for pattern in patterns]

    def matches(relative: str) -> bool:
        return any(regex.match(relative) for regex in compiled)

    return matches


def _is_hidden(relative: str) -> bool:
    return any(part.startswith(".") for part in relative.split("/"))


@dataclass(frozen=True)
class SelectedInput:
    relative_path: str  # normalized POSIX relative path
    absolute_path: Path


def select_files(root: Path, config: InputConfig) -> list[SelectedInput]:
    """Recursive deterministic selection under root (spec 18.9-18.11)."""
    include = _build_matcher(config.include)
    exclude = _build_matcher(config.exclude)
    selected: list[SelectedInput] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            absolute = Path(dirpath) / filename
            relative = absolute.relative_to(root).as_posix()
            if not config.include_hidden and _is_hidden(relative):
                continue
            if not include(relative):
                continue
            if exclude(relative):
                continue
            selected.append(SelectedInput(relative_path=relative, absolute_path=absolute))
    selected.sort(key=lambda item: item.relative_path)
    if len(selected) > config.max_files:
        raise ResourceLimitError(
            f"directory selection exceeds max_files ({len(selected)} > {config.max_files}) "
            "(DIRECTORY_TOO_MANY_FILES)"
        )
    total = 0
    for item in selected:
        if item.absolute_path.is_symlink():
            continue  # skipped with diagnostic by the caller layer
        total += item.absolute_path.stat().st_size
    if total > config.max_total_bytes:
        raise ResourceLimitError(
            f"directory selection exceeds max_total_bytes ({total}) (DIRECTORY_TOO_MANY_BYTES)"
        )
    return selected
