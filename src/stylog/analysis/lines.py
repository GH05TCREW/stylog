"""Physical line scanning and paragraph segmentation (spec 7.9, 8.2).

Line-break sequences: CRLF (one break), LF, CR, U+2028, U+2029. A physical
line is the characters between break sequences. For nonempty source the line
count is ``number_of_line_break_sequences + 1``; the final (possibly empty)
line after a trailing break is represented.
"""

from __future__ import annotations

from dataclasses import dataclass

from stylog.analysis.whitespace import is_white_space

END_LF = "lf"
END_CRLF = "crlf"
END_CR = "cr"
END_LINE_SEPARATOR = "line_separator"
END_PARAGRAPH_SEPARATOR = "paragraph_separator"

LINE_SEPARATOR = " "  # terminates a physical line, not a paragraph by itself
PARAGRAPH_SEPARATOR = " "  # always terminates the current paragraph


@dataclass(frozen=True)
class PhysicalLine:
    content: str  # without the line-ending sequence
    ending: str | None  # None for the final line when the text has no trailing break
    row: int  # 1-based physical row


def scan_lines(text: str) -> list[PhysicalLine]:
    """Split text into physical lines (spec 8.2 semantics)."""
    if text == "":
        return []
    lines: list[PhysicalLine] = []
    start = 0
    row = 1
    index = 0
    n = len(text)
    while index < n:
        char = text[index]
        if char == "\r":
            if index + 1 < n and text[index + 1] == "\n":
                lines.append(PhysicalLine(text[start:index], END_CRLF, row))
                index += 2
            else:
                lines.append(PhysicalLine(text[start:index], END_CR, row))
                index += 1
            row += 1
            start = index
        elif char == "\n":
            lines.append(PhysicalLine(text[start:index], END_LF, row))
            index += 1
            row += 1
            start = index
        elif char == LINE_SEPARATOR:
            lines.append(PhysicalLine(text[start:index], END_LINE_SEPARATOR, row))
            index += 1
            row += 1
            start = index
        elif char == PARAGRAPH_SEPARATOR:
            lines.append(PhysicalLine(text[start:index], END_PARAGRAPH_SEPARATOR, row))
            index += 1
            row += 1
            start = index
        else:
            index += 1
    lines.append(PhysicalLine(text[start:], None, row))
    return lines


def count_line_breaks(text: str) -> dict[str, int]:
    """Counts of each line-ending sequence category over the whole text."""
    counts = {
        END_LF: 0,
        END_CRLF: 0,
        END_CR: 0,
        END_LINE_SEPARATOR: 0,
        END_PARAGRAPH_SEPARATOR: 0,
    }
    for line in scan_lines(text):
        if line.ending is not None:
            counts[line.ending] += 1
    return counts


def is_blank(content: str) -> bool:
    """Blank line: zero chars or only Unicode White_Space code points."""
    return all(is_white_space(char) for char in content)


def segment_paragraphs(lines: list[PhysicalLine]) -> list[list[PhysicalLine]]:
    """Paragraphs per spec 7.9.

    A paragraph is a maximal run of one or more nonblank physical lines. One
    or more blank lines separate paragraphs. U+2029 always terminates the
    current paragraph. U+2028 ends a line without forcing a paragraph break.
    """
    paragraphs: list[list[PhysicalLine]] = []
    current: list[PhysicalLine] = []
    for line in lines:
        if is_blank(line.content):
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(line)
        if line.ending == END_PARAGRAPH_SEPARATOR:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs
