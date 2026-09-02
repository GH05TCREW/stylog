"""Pinned Unicode 17.0 White_Space table and classification (spec 7.2).

This list is a versioned scientific resource: exactly the 25 code points of
the Unicode 17.0 White_Space property. ``str.isspace()`` is never used.
"""

from __future__ import annotations

WHITE_SPACE_CODEPOINTS: frozenset[int] = frozenset(
    [
        *range(0x0009, 0x000E),  # U+0009-U+000D
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        *range(0x2000, 0x200B),  # U+2000-U+200A
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    ]
)

assert len(WHITE_SPACE_CODEPOINTS) == 25

WHITE_SPACE_CHARS: frozenset[str] = frozenset(chr(cp) for cp in WHITE_SPACE_CODEPOINTS)

_WHITESPACE_CLASS_BY_CODEPOINT = {
    0x0020: "space_ascii",
    0x0009: "tab",
    0x000A: "line_feed",
    0x000D: "carriage_return",
    0x2028: "line_separator",
    0x2029: "paragraph_separator",
}


def is_white_space(char: str) -> bool:
    return ord(char) in WHITE_SPACE_CODEPOINTS


def whitespace_class(char: str) -> str:
    """Classify one White_Space code point (spec 7.2 categories)."""
    return _WHITESPACE_CLASS_BY_CODEPOINT.get(ord(char), "other_white_space")
