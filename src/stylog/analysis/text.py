"""Deterministic text analysis core (spec sections 6.2-6.5, 7, 10).

Pure tokenization, sentence segmentation, and marker counting plus the five
text analyzers (sample, surface, lexical, structure, English function words).
No file IO, no network, no normalization: every algorithm operates directly
on the decoded code-point sequence.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from stylog.analysis import build, lines, stats
from stylog.analysis.base import TEXT_BACKEND, AnalyzerOutput, BaseAnalyzer, all_status_observations
from stylog.analysis.registry import (
    ANALYZER_TEXT_FUNCTION_WORDS_EN,
    ANALYZER_TEXT_LEXICAL,
    ANALYZER_TEXT_SAMPLE,
    ANALYZER_TEXT_STRUCTURE,
    ANALYZER_TEXT_SURFACE,
    features_owned_by,
)
from stylog.analysis.whitespace import is_white_space, whitespace_class
from stylog.domain.diagnostic import Diagnostic, DiagnosticSeverity, make_diagnostic
from stylog.domain.feature import FeatureObservation, FeatureStatus
from stylog.domain.provenance import ResourceSignature
from stylog.runtime import AnalysisContext, RuntimeArtifact

# ---------------------------------------------------------------------------
# Tokenizer (spec 7.4, tokenizer id stylog.text.tokenizer/1.0.0)
# ---------------------------------------------------------------------------

TokenKind = Literal["word", "number"]


@dataclass(frozen=True)
class TextToken:
    """One lexical token with code-point offsets into the source text."""

    kind: TokenKind
    text: str
    start: int  # inclusive code-point offset
    end: int  # exclusive code-point offset


_APOSTROPHES = frozenset({"'", "’"})  # U+0027, U+2019
_NUMBER_SEPARATORS = frozenset(".,_")


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


def _is_letter_or_mark(char: str) -> bool:
    return unicodedata.category(char)[0] in ("L", "M")


def _is_decimal_digit(char: str) -> bool:
    return unicodedata.category(char) == "Nd"


def tokenize_text(text: str) -> list[TextToken]:
    """Tokenize into WORD and NUMBER tokens (spec 7.4).

    WORD begins with L*, continues through L*/M* and internal apostrophes
    (U+0027/U+2019, kept only when preceded inside the same WORD by L/M and
    immediately followed by an L code point). NUMBER begins with Nd and
    continues through Nd plus ".", ",", "_" separators, each kept only when
    immediately preceded and followed by Nd. Everything else is not a token.
    """
    tokens: list[TextToken] = []
    n = len(text)
    index = 0
    while index < n:
        char = text[index]
        if _is_letter(char):
            start = index
            index += 1
            while index < n:
                current = text[index]
                if _is_letter_or_mark(current) or (
                    current in _APOSTROPHES
                    and _is_letter_or_mark(text[index - 1])
                    and index + 1 < n
                    and _is_letter(text[index + 1])
                ):
                    index += 1
                else:
                    break
            tokens.append(TextToken("word", text[start:index], start, index))
        elif _is_decimal_digit(char):
            start = index
            index += 1
            while index < n:
                current = text[index]
                if _is_decimal_digit(current) or (
                    current in _NUMBER_SEPARATORS
                    and _is_decimal_digit(text[index - 1])
                    and index + 1 < n
                    and _is_decimal_digit(text[index + 1])
                ):
                    index += 1
                else:
                    break
            tokens.append(TextToken("number", text[start:index], start, index))
        else:
            index += 1
    return tokens


# ---------------------------------------------------------------------------
# Sentence segmentation (spec 7.8, segmenter id stylog.text.sentence_segmenter/1.0.0)
# ---------------------------------------------------------------------------

SENTENCE_TERMINALS = frozenset(
    [".", "!", "?", "…", "。", "！", "？"]  # U+002E U+0021 U+003F U+2026 U+3002 U+FF01 U+FF1F
)
SENTENCE_CLOSERS = frozenset(
    [
        "'",  # U+0027
        '"',  # U+0022
        ")",
        "]",
        "}",
        "’",  # U+2019
        "”",  # U+201D
        "»",  # U+00BB
        "›",  # U+203A
        "」",  # U+300D
        "』",  # U+300F
        "】",  # U+3011
    ]
)


def _is_terminal(text: str, index: int) -> bool:
    char = text[index]
    if char not in SENTENCE_TERMINALS:
        return False
    # Decimal rule: U+002E between two Nd code points is not terminal.
    return not (
        char == "."
        and index > 0
        and index + 1 < len(text)
        and _is_decimal_digit(text[index - 1])
        and _is_decimal_digit(text[index + 1])
    )


def _trim_white_space(span: str) -> str:
    start = 0
    end = len(span)
    while start < end and is_white_space(span[start]):
        start += 1
    while end > start and is_white_space(span[end - 1]):
        end -= 1
    return span[start:end]


def segment_sentences(paragraph_text: str) -> list[str]:
    """Segment one paragraph's text into sentences (spec 7.8).

    A boundary follows a maximal terminal cluster plus any closers when the
    next code point is White_Space or the paragraph ends. Sentence spans are
    trimmed of leading/trailing White_Space. Any non-whitespace residual text
    forms a sentence. No abbreviation dictionary.
    """
    sentences: list[str] = []
    n = len(paragraph_text)
    segment_start = 0
    index = 0
    while index < n:
        if not _is_terminal(paragraph_text, index):
            index += 1
            continue
        cluster_end = index + 1
        while cluster_end < n and _is_terminal(paragraph_text, cluster_end):
            cluster_end += 1
        span_end = cluster_end
        while span_end < n and paragraph_text[span_end] in SENTENCE_CLOSERS:
            span_end += 1
        if span_end == n or is_white_space(paragraph_text[span_end]):
            sentence = _trim_white_space(paragraph_text[segment_start:span_end])
            if sentence:
                sentences.append(sentence)
            segment_start = span_end
        index = span_end
    residual = _trim_white_space(paragraph_text[segment_start:])
    if residual:
        sentences.append(residual)
    return sentences


# ---------------------------------------------------------------------------
# Marker events (spec 7.7)
# ---------------------------------------------------------------------------

_MARKER_BY_CHAR = {
    "'": "apostrophe_ascii",
    "’": "apostrophe_right",
    '"': "quote_ascii",
    "‘": "quote_left_single",
    "“": "quote_left_double",
    "”": "quote_right_double",
    "«": "guillemet_left",
    "»": "guillemet_right",
    "-": "hyphen_minus",
    "–": "en_dash",
    "—": "em_dash",
    "…": "horizontal_ellipsis",
}
ASCII_THREE_DOTS = "ascii_three_dots"


def count_markers(text: str) -> dict[str, int]:
    """Count the 13 marker events (spec 7.7).

    Twelve single-code-point markers count per occurrence;
    ``ascii_three_dots`` counts literal "..." left-to-right non-overlapping.
    """
    counts: dict[str, int] = {}
    index = 0
    n = len(text)
    while index < n:
        if text.startswith("...", index):
            counts[ASCII_THREE_DOTS] = counts.get(ASCII_THREE_DOTS, 0) + 1
            index += 3
            continue
        marker = _MARKER_BY_CHAR.get(text[index])
        if marker is not None:
            counts[marker] = counts.get(marker, 0) + 1
        index += 1
    return counts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_LETTER_CASE_BY_CATEGORY = {"Lu": "upper", "Ll": "lower", "Lt": "title"}


def _punctuation_key(char: str) -> str:
    code_point = ord(char)
    if code_point <= 0xFFFF:
        return f"U+{code_point:04X}"
    return f"U+{code_point:06X}"


_ENDING_LENGTHS = {
    lines.END_LF: 1,
    lines.END_CRLF: 2,
    lines.END_CR: 1,
    lines.END_LINE_SEPARATOR: 1,
    lines.END_PARAGRAPH_SEPARATOR: 1,
    None: 0,
}


def _paragraph_texts(
    text: str,
    physical_lines: list[lines.PhysicalLine],
    paragraphs: list[list[lines.PhysicalLine]],
) -> list[str]:
    """Exact paragraph slices, internal line-ending code points preserved.

    Spans are derived from ``scan_lines`` offsets over every physical line
    (blank separator lines included), keyed by each line's unique row.
    """
    span_by_row: dict[int, tuple[int, int]] = {}
    position = 0
    for line in physical_lines:
        start = position
        end = start + len(line.content)
        span_by_row[line.row] = (start, end)
        position = end + _ENDING_LENGTHS[line.ending]
    return [
        text[span_by_row[paragraph[0].row][0] : span_by_row[paragraph[-1].row][1]]
        for paragraph in paragraphs
    ]


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------


class TextSampleAnalyzer(BaseAnalyzer):
    """Raw byte and code-point counts (spec 6.2). Always ok, even when 0."""

    analyzer_id = ANALYZER_TEXT_SAMPLE
    backend = TEXT_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        values = {
            "text.sample.byte_count": len(artifact.raw_bytes),
            "text.sample.character_count": len(artifact.text),
        }
        observations = tuple(
            build.ok(
                fdef,
                self.analyzer_id,
                self.implementation_version,
                build.int_value(values[fdef.feature_id]),
                1,
            )
            for fdef in features_owned_by(self.analyzer_id)
        )
        return AnalyzerOutput(observations=observations)


class TextSurfaceAnalyzer(BaseAnalyzer):
    """Surface categorical distributions (spec 6.2, 7.2, 7.3, 7.7, 7.10)."""

    analyzer_id = ANALYZER_TEXT_SURFACE
    backend = TEXT_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        text = artifact.text
        line_ending_counts = lines.count_line_breaks(text)
        whitespace_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        case_counts: Counter[str] = Counter()
        punctuation_counts: Counter[str] = Counter()
        for char in text:
            category = unicodedata.category(char)
            category_counts[category] += 1
            if is_white_space(char):
                whitespace_counts[whitespace_class(char)] += 1
            if category.startswith("L"):
                case_counts[_LETTER_CASE_BY_CATEGORY.get(category, "uncased")] += 1
            elif category.startswith("P"):
                punctuation_counts[_punctuation_key(char)] += 1
        distributions = {
            "text.surface.letter_case": case_counts,
            "text.surface.line_ending": line_ending_counts,
            "text.surface.marker_style": count_markers(text),
            "text.surface.punctuation_codepoint": punctuation_counts,
            "text.surface.unicode_general_category": category_counts,
            "text.surface.whitespace_class": whitespace_counts,
        }
        observations = tuple(
            build.categorical_observation(
                fdef,
                self.analyzer_id,
                self.implementation_version,
                distributions[fdef.feature_id],
            )
            for fdef in features_owned_by(self.analyzer_id)
        )
        return AnalyzerOutput(observations=observations)


class TextLexicalAnalyzer(BaseAnalyzer):
    """Lexical tokens, types, TTR, entropy (spec 6.3, 7.4-7.6, 7.11-7.13)."""

    analyzer_id = ANALYZER_TEXT_LEXICAL
    backend = TEXT_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        tokens = tokenize_text(artifact.text)
        words = [token for token in tokens if token.kind == "word"]
        numbers = [token for token in tokens if token.kind == "number"]
        word_count = len(words)
        casefold_counts: Counter[str] = Counter(token.text.casefold() for token in words)
        type_count = len(casefold_counts)
        hapax_count = sum(1 for count in casefold_counts.values() if count == 1)
        window_ttr_enabled = ctx.config.analysis.text.window_ttr_100

        observations: list[FeatureObservation] = []
        for fdef in features_owned_by(self.analyzer_id):
            feature_id = fdef.feature_id
            if feature_id == "text.lexical.word_count":
                observation = build.ok(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    build.int_value(word_count),
                    1,
                )
            elif feature_id == "text.lexical.number_count":
                observation = build.ok(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    build.int_value(len(numbers)),
                    1,
                )
            elif feature_id == "text.lexical.token_kind":
                observation = build.categorical_observation(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    {"word": word_count, "number": len(numbers)},
                )
            elif feature_id == "text.lexical.word_length":
                observation = build.histogram_observation(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    (len(token.text) for token in words),
                )
            elif feature_id == "text.lexical.type_count_casefold":
                observation = build.ok(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    build.int_value(type_count),
                    word_count,
                )
            elif feature_id == "text.lexical.ttr_casefold":
                observation = self._ratio_observation(fdef, type_count, word_count)
            elif feature_id == "text.lexical.hapax_type_count_casefold":
                observation = build.ok(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    build.int_value(hapax_count),
                    word_count,
                )
            elif feature_id == "text.lexical.hapax_token_share_casefold":
                observation = self._ratio_observation(fdef, hapax_count, word_count)
            elif feature_id == "text.lexical.window_ttr_100":
                observation = self._window_ttr_observation(fdef, words, window_ttr_enabled)
            elif feature_id == "text.lexical.word_entropy_bits_casefold":
                observation = self._float_observation(
                    fdef,
                    stats.shannon_entropy_bits(casefold_counts.values(), word_count)
                    if word_count
                    else None,
                    word_count,
                )
            elif feature_id == "text.lexical.word_simpson_concentration_casefold":
                observation = self._float_observation(
                    fdef,
                    stats.simpson_concentration(casefold_counts.values(), word_count)
                    if word_count
                    else None,
                    word_count,
                )
            else:  # pragma: no cover - registry drift guard
                raise KeyError(feature_id)
            observations.append(observation)
        return AnalyzerOutput(observations=tuple(observations))

    def _ratio_observation(self, fdef, numerator: int, denominator: int) -> FeatureObservation:
        return build.ratio_observation(
            fdef, self.analyzer_id, self.implementation_version, numerator, denominator
        )

    def _float_observation(self, fdef, value: float | None, support: int) -> FeatureObservation:
        if value is None:
            return build.status(
                fdef, self.analyzer_id, self.implementation_version,
                FeatureStatus.INSUFFICIENT_SUPPORT,
            )
        return build.ok(
            fdef,
            self.analyzer_id,
            self.implementation_version,
            build.float_value(value),
            support,
        )

    def _window_ttr_observation(
        self, fdef, words: list[TextToken], enabled: bool
    ) -> FeatureObservation:
        if not enabled:
            return build.status(
                fdef, self.analyzer_id, self.implementation_version, FeatureStatus.DISABLED
            )
        word_count = len(words)
        if word_count < 100:
            return build.status(
                fdef,
                self.analyzer_id,
                self.implementation_version,
                FeatureStatus.INSUFFICIENT_SUPPORT,
            )
        window_ttrs = [
            len({token.text.casefold() for token in words[start : start + 100]}) / 100
            for start in range(0, word_count - 99, 100)
        ]
        value = build.summary_value(window_ttrs)
        return build.ok(
            fdef,
            self.analyzer_id,
            self.implementation_version,
            value,
            len(window_ttrs),
        )


class TextStructureAnalyzer(BaseAnalyzer):
    """Sentence and paragraph structure (spec 6.4, 7.8, 7.9)."""

    analyzer_id = ANALYZER_TEXT_STRUCTURE
    backend = TEXT_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        text = artifact.text
        physical_lines = lines.scan_lines(text)
        paragraphs = lines.segment_paragraphs(physical_lines)
        paragraph_texts = _paragraph_texts(text, physical_lines, paragraphs)
        paragraph_count = len(paragraphs)

        sentence_token_lengths: list[int] = []
        sentence_char_lengths: list[int] = []
        paragraph_sentence_counts: list[int] = []
        paragraph_token_counts: list[int] = []
        for paragraph_text in paragraph_texts:
            sentences = segment_sentences(paragraph_text)
            paragraph_sentence_counts.append(len(sentences))
            paragraph_token_counts.append(len(tokenize_text(paragraph_text)))
            for sentence in sentences:
                sentence_token_lengths.append(len(tokenize_text(sentence)))
                sentence_char_lengths.append(len(sentence))
        sentence_count = len(sentence_char_lengths)

        histograms = {
            "text.structure.paragraph_sentence_count": paragraph_sentence_counts,
            "text.structure.paragraph_token_count": paragraph_token_counts,
            "text.structure.sentence_length_characters": sentence_char_lengths,
            "text.structure.sentence_length_tokens": sentence_token_lengths,
        }
        counts = {
            "text.structure.paragraph_count": paragraph_count,
            "text.structure.sentence_count": sentence_count,
        }
        observations: list[FeatureObservation] = []
        for fdef in features_owned_by(self.analyzer_id):
            feature_id = fdef.feature_id
            if feature_id in counts:
                observation = build.ok(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    build.int_value(counts[feature_id]),
                    1,
                )
            else:
                observation = build.histogram_observation(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    histograms[feature_id],
                )
            observations.append(observation)
        return AnalyzerOutput(observations=tuple(observations))


class TextFunctionWordsEnAnalyzer(BaseAnalyzer):
    """English function-word share and distribution (spec 6.5)."""

    analyzer_id = ANALYZER_TEXT_FUNCTION_WORDS_EN
    backend = TEXT_BACKEND

    def resources(self, ctx: AnalysisContext) -> tuple[ResourceSignature, ...]:
        signature = ctx.resources.function_words_en_signature
        if signature is None:
            return ()
        return (signature,)

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        fdefs = features_owned_by(self.analyzer_id)
        if not ctx.config.analysis.text.function_words_en:
            return self._all_status(FeatureStatus.DISABLED)
        if artifact.language == "und":
            diagnostic = make_diagnostic(
                "LANGUAGE_UNSPECIFIED",
                DiagnosticSeverity.WARNING,
                analyzer_id=self.analyzer_id,
            )
            output = self._all_status(FeatureStatus.UNAVAILABLE, (diagnostic,))
            return output
        if artifact.language != "en":
            return self._all_status(FeatureStatus.NOT_APPLICABLE)
        lexemes = ctx.resources.function_words_en
        if lexemes is None:
            return self._all_status(FeatureStatus.UNAVAILABLE)

        words = [
            token for token in tokenize_text(artifact.text) if token.kind == "word"
        ]
        word_count = len(words)
        matched = [token.text.casefold() for token in words if token.text.casefold() in lexemes]
        matched_counts: Counter[str] = Counter(matched)

        observations: list[FeatureObservation] = []
        for fdef in fdefs:
            feature_id = fdef.feature_id
            if feature_id == "text.function_words.en.token_share":
                observation = build.ratio_observation(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    len(matched),
                    word_count,
                )
            elif feature_id == "text.function_words.en.lexeme_distribution":
                observation = build.categorical_observation(
                    fdef,
                    self.analyzer_id,
                    self.implementation_version,
                    matched_counts,
                )
            else:  # pragma: no cover - registry drift guard
                raise KeyError(feature_id)
            observations.append(observation)
        return AnalyzerOutput(observations=tuple(observations))

    def _all_status(
        self,
        status: FeatureStatus,
        diagnostics: tuple[Diagnostic, ...] = (),
    ) -> AnalyzerOutput:
        observations = all_status_observations(
            self.analyzer_id, self.implementation_version, status
        )
        return AnalyzerOutput(observations=observations, diagnostics=diagnostics)
