"""Identifier case-style algorithm (spec 8.9).

Shared by the Python analyzer and the tree-sitter analyzer. All classification
is by Unicode general category, never by ``str.isupper()``/``str.islower()``
alone on whole strings.
"""

from __future__ import annotations

import re
import unicodedata

_DUNDER_RE = re.compile(r"^__[A-Za-z0-9_]+__$")
_UNDERSCORE_RUN_RE = re.compile(r"_+")


def _category(char: str) -> str:
    return unicodedata.category(char)


def _is_lower_letter(char: str) -> bool:
    return _category(char) == "Ll"


def _is_upper_letter(char: str) -> bool:
    return _category(char) == "Lu"


def _is_cased_letter(char: str) -> bool:
    return _category(char) in ("Lu", "Ll", "Lt")


def split_components(name: str) -> list[str]:
    """Underscore split then case-transition split (spec 8.9 steps A and B)."""
    parts = [part for part in _UNDERSCORE_RUN_RE.split(name) if part]
    if not parts:
        return []
    components: list[str] = []
    for part in parts:
        start = 0
        for index in range(1, len(part)):
            previous = part[index - 1]
            current = part[index]
            following = part[index + 1] if index + 1 < len(part) else ""
            rule1 = (_is_lower_letter(previous) or _category(previous) == "Nd") and (
                _is_upper_letter(current)
            )
            rule2 = (
                _is_upper_letter(previous)
                and _is_upper_letter(current)
                and following != ""
                and _is_lower_letter(following)
            )
            if rule1 or rule2:
                components.append(part[start:index])
                start = index
        components.append(part[start:])
    return [component for component in components if component]


def classify_style(name: str) -> str:
    """Classify an identifier per spec 8.9.

    Returns one of: discard, dunder, snake_lower, snake_upper, camel_lower,
    pascal, lower, upper, mixed, uncased, other.
    """
    if name == "_":
        return "discard"
    if len(name) > 4 and _DUNDER_RE.match(name):
        return "dunder"
    if not split_components(name):
        return "other"
    core = name.strip("_")
    cased = [char for char in core if _is_cased_letter(char)]
    if "_" in core:
        if cased and all(_is_lower_letter(char) for char in cased):
            return "snake_lower"
        if cased and all(_is_upper_letter(char) for char in cased):
            return "snake_upper"
        return "mixed" if cased else "uncased"
    if not cased:
        return "uncased"
    if all(_is_lower_letter(char) for char in cased):
        return "lower"
    if all(_is_upper_letter(char) for char in cased):
        return "upper"
    first_cased = cased[0]
    later = cased[1:]
    if _is_lower_letter(first_cased) and any(_is_upper_letter(char) for char in later):
        return "camel_lower"
    if _is_upper_letter(first_cased) and any(_is_lower_letter(char) for char in later):
        return "pascal"
    return "mixed"
