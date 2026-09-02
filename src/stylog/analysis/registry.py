"""The Stylog v0.1 feature registry (spec section 6).

Registry version ``stylog.features/1.0.0``; every feature has semantic version
``1.0.0``. This table is the single source of truth for feature ownership,
geometry, reducer, comparison metric, support kind, top-codes, and confound
tags. It is internal runtime data, not a portable artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stylog.domain.evidence import AggregationKind

FEATURE_REGISTRY_VERSION = "1.0.0"
FEATURE_REGISTRY_ID = f"stylog.features/{FEATURE_REGISTRY_VERSION}"
FEATURE_SEMANTIC_VERSION = "1.0.0"

# Analyzer identifiers and implementation versions.
ANALYZER_TEXT_SAMPLE = "stylog.text.sample"
ANALYZER_TEXT_SURFACE = "stylog.text.surface"
ANALYZER_TEXT_LEXICAL = "stylog.text.lexical"
ANALYZER_TEXT_STRUCTURE = "stylog.text.structure"
ANALYZER_TEXT_FUNCTION_WORDS_EN = "stylog.text.function_words.en"
ANALYZER_TEXT_LINGUISTIC = "stylog.text.linguistic"
ANALYZER_CODE_SAMPLE = "stylog.code.sample"
ANALYZER_CODE_SURFACE = "stylog.code.surface"
ANALYZER_CODE_PYTHON_TOKENS = "stylog.code.python.tokens"
ANALYZER_CODE_PYTHON_AST = "stylog.code.python.ast"
ANALYZER_CODE_TREE_SITTER = "stylog.code.tree_sitter"

ANALYZER_IMPLEMENTATION_VERSION = "1.0.0"

FUNCTION_WORDS_EN_RESOURCE_ID = "stylog.function_words.en"
FUNCTION_WORDS_EN_RESOURCE_VERSION = "1.0.0"
FUNCTION_WORDS_EN_SHA256 = "2177da3067b27ab7f1c1c228474bd5f3c6d59d3c71ffac6ab52c787c4ea881f5"


@dataclass(frozen=True)
class FeatureDef:
    feature_id: str
    analyzer_id: str
    geometry: str  # integer|float|ratio|categorical_distribution|ordered_histogram|summary
    reducer: AggregationKind
    metric: str  # ABS|SPD|JSD2|W1|NONE
    support_kind: str
    top_code: int | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    runtime_sensitive: bool = False
    resource_ids: tuple[str, ...] = ()

    @property
    def family(self) -> str:
        return self.feature_id.rsplit(".", 1)[0]


def _d(
    feature_id: str,
    analyzer_id: str,
    geometry: str,
    reducer: AggregationKind,
    metric: str,
    support_kind: str,
    top_code: int | None = None,
    tags: frozenset[str] = frozenset(),
    runtime_sensitive: bool = False,
    resource_ids: tuple[str, ...] = (),
) -> FeatureDef:
    return FeatureDef(
        feature_id=feature_id,
        analyzer_id=analyzer_id,
        geometry=geometry,
        reducer=reducer,
        metric=metric,
        support_kind=support_kind,
        top_code=top_code,
        tags=tags,
        runtime_sensitive=runtime_sensitive,
        resource_ids=resource_ids,
    )


SUM = AggregationKind.EXACT_SUM
RATIO = AggregationKind.RATIO_POOL
CAT = AggregationKind.CATEGORICAL_POOL
HIST = AggregationKind.HISTOGRAM_POOL
SAMPLE = AggregationKind.SAMPLE_SUMMARY
NONE = AggregationKind.NOT_AGGREGATABLE

_FW = (FUNCTION_WORDS_EN_RESOURCE_ID,)

FEATURES: dict[str, FeatureDef] = {
    d.feature_id: d
    for d in [
        # --- text sample/surface (6.2) ---
        _d("text.sample.byte_count", ANALYZER_TEXT_SAMPLE, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive"})),
        _d("text.sample.character_count", ANALYZER_TEXT_SAMPLE, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive", "unicode_version_sensitive"}), runtime_sensitive=True),
        _d("text.surface.line_ending", ANALYZER_TEXT_SURFACE, "categorical_distribution", CAT, "JSD2",
           "line break", tags=frozenset({"surface", "toolchain_sensitive"})),
        _d("text.surface.whitespace_class", ANALYZER_TEXT_SURFACE, "categorical_distribution", CAT,
           "JSD2", "whitespace code point", tags=frozenset({"surface", "toolchain_sensitive"})),
        _d("text.surface.unicode_general_category", ANALYZER_TEXT_SURFACE, "categorical_distribution",
           CAT, "JSD2", "character",
           tags=frozenset({"unicode_version_sensitive", "content_sensitive"}), runtime_sensitive=True),
        _d("text.surface.letter_case", ANALYZER_TEXT_SURFACE, "categorical_distribution", CAT, "JSD2",
           "Unicode letter", tags=frozenset({"surface", "language_specific"})),
        _d("text.surface.punctuation_codepoint", ANALYZER_TEXT_SURFACE, "categorical_distribution",
           CAT, "JSD2", "punctuation code point",
           tags=frozenset({"surface", "content_sensitive"})),
        _d("text.surface.marker_style", ANALYZER_TEXT_SURFACE, "categorical_distribution", CAT,
           "JSD2", "marker event", tags=frozenset({"surface", "toolchain_sensitive"})),
        # --- text lexical (6.3) ---
        _d("text.lexical.word_count", ANALYZER_TEXT_LEXICAL, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive"})),
        _d("text.lexical.number_count", ANALYZER_TEXT_LEXICAL, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive", "content_sensitive"})),
        _d("text.lexical.token_kind", ANALYZER_TEXT_LEXICAL, "categorical_distribution", CAT, "JSD2",
           "lexical token", tags=frozenset({"content_sensitive"})),
        _d("text.lexical.word_length", ANALYZER_TEXT_LEXICAL, "ordered_histogram", HIST, "W1", "word",
           top_code=31, tags=frozenset({"language_specific", "length_sensitive"})),
        _d("text.lexical.type_count_casefold", ANALYZER_TEXT_LEXICAL, "integer", SAMPLE, "NONE",
           "word", tags=frozenset({"length_sensitive", "content_sensitive",
                                   "unicode_version_sensitive"}), runtime_sensitive=True),
        _d("text.lexical.ttr_casefold", ANALYZER_TEXT_LEXICAL, "ratio", SAMPLE, "ABS", "word",
           tags=frozenset({"length_sensitive", "content_sensitive",
                           "unicode_version_sensitive"}), runtime_sensitive=True),
        _d("text.lexical.hapax_type_count_casefold", ANALYZER_TEXT_LEXICAL, "integer", SAMPLE,
           "NONE", "word", tags=frozenset({"length_sensitive", "content_sensitive",
                                           "unicode_version_sensitive"}), runtime_sensitive=True),
        _d("text.lexical.hapax_token_share_casefold", ANALYZER_TEXT_LEXICAL, "ratio", SAMPLE, "ABS",
           "word", tags=frozenset({"length_sensitive", "content_sensitive",
                                   "unicode_version_sensitive"}), runtime_sensitive=True),
        _d("text.lexical.window_ttr_100", ANALYZER_TEXT_LEXICAL, "summary", NONE, "NONE",
           "100-word window", tags=frozenset({"content_sensitive", "unicode_version_sensitive"}),
           runtime_sensitive=True),
        _d("text.lexical.word_entropy_bits_casefold", ANALYZER_TEXT_LEXICAL, "float", SAMPLE,
           "NONE", "word", tags=frozenset({"content_sensitive", "length_sensitive"})),
        _d("text.lexical.word_simpson_concentration_casefold", ANALYZER_TEXT_LEXICAL, "float",
           SAMPLE, "ABS", "word", tags=frozenset({"content_sensitive", "length_sensitive"})),
        # --- text structure (6.4) ---
        _d("text.structure.sentence_count", ANALYZER_TEXT_STRUCTURE, "integer", SUM, "NONE",
           "artifact", tags=frozenset({"length_sensitive", "language_specific"})),
        _d("text.structure.sentence_length_tokens", ANALYZER_TEXT_STRUCTURE, "ordered_histogram",
           HIST, "W1", "sentence", top_code=101,
           tags=frozenset({"language_specific", "content_sensitive"})),
        _d("text.structure.sentence_length_characters", ANALYZER_TEXT_STRUCTURE,
           "ordered_histogram", HIST, "W1", "sentence", top_code=501,
           tags=frozenset({"language_specific", "content_sensitive"})),
        _d("text.structure.paragraph_count", ANALYZER_TEXT_STRUCTURE, "integer", SUM, "NONE",
           "artifact", tags=frozenset({"surface", "length_sensitive"})),
        _d("text.structure.paragraph_sentence_count", ANALYZER_TEXT_STRUCTURE,
           "ordered_histogram", HIST, "W1", "paragraph", top_code=51,
           tags=frozenset({"surface", "language_specific"})),
        _d("text.structure.paragraph_token_count", ANALYZER_TEXT_STRUCTURE, "ordered_histogram",
           HIST, "W1", "paragraph", top_code=501,
           tags=frozenset({"surface", "content_sensitive"})),
        # --- English function words (6.5) ---
        _d("text.function_words.en.token_share", ANALYZER_TEXT_FUNCTION_WORDS_EN, "ratio", RATIO,
           "ABS", "word", tags=frozenset({"language_specific", "content_reduced"}),
           resource_ids=_FW),
        _d("text.function_words.en.lexeme_distribution", ANALYZER_TEXT_FUNCTION_WORDS_EN,
           "categorical_distribution", CAT, "JSD2", "matched function word",
           tags=frozenset({"language_specific", "content_reduced", "resource_sensitive"}),
           resource_ids=_FW),
        # --- generic code (6.6) ---
        _d("code.sample.byte_count", ANALYZER_CODE_SAMPLE, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive"})),
        _d("code.sample.character_count", ANALYZER_CODE_SAMPLE, "integer", SUM, "NONE", "artifact",
           tags=frozenset({"length_sensitive"})),
        _d("code.sample.physical_line_count", ANALYZER_CODE_SAMPLE, "integer", SUM, "NONE",
           "artifact", tags=frozenset({"length_sensitive"})),
        _d("code.surface.line_ending", ANALYZER_CODE_SURFACE, "categorical_distribution", CAT,
           "JSD2", "line break", tags=frozenset({"surface", "toolchain_sensitive"})),
        _d("code.surface.whitespace_class", ANALYZER_CODE_SURFACE, "categorical_distribution", CAT,
           "JSD2", "whitespace code point", tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.indent_kind", ANALYZER_CODE_SURFACE, "categorical_distribution", CAT,
           "JSD2", "nonblank line", tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.indent_char_count", ANALYZER_CODE_SURFACE, "ordered_histogram", HIST,
           "W1", "nonblank line", top_code=33,
           tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.nonblank_line_length", ANALYZER_CODE_SURFACE, "ordered_histogram", HIST,
           "W1", "nonblank line", top_code=201,
           tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.blank_line_share", ANALYZER_CODE_SURFACE, "ratio", RATIO, "ABS",
           "physical line", tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.blank_run_length", ANALYZER_CODE_SURFACE, "ordered_histogram", HIST, "W1",
           "blank run", top_code=11, tags=frozenset({"surface", "formatter_sensitive"})),
        _d("code.surface.trailing_space_line_share", ANALYZER_CODE_SURFACE, "ratio", RATIO, "ABS",
           "physical line", tags=frozenset({"surface", "formatter_sensitive"})),
        # --- Python lexical (6.7) ---
        _d("code.python.lexical.token_class", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "token",
           tags=frozenset({"runtime_sensitive", "task_sensitive"}), runtime_sensitive=True),
        _d("code.python.lexical.keyword_distribution", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "token",
           tags=frozenset({"runtime_sensitive", "task_sensitive"}), runtime_sensitive=True),
        _d("code.python.lexical.operator_distribution", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "token", tags=frozenset({"task_sensitive"})),
        _d("code.python.lexical.number_form", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "token", tags=frozenset({"task_sensitive"})),
        _d("code.python.lexical.string_quote", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "string token",
           tags=frozenset({"surface", "formatter_sensitive", "runtime_sensitive"}),
           runtime_sensitive=True),
        _d("code.python.lexical.string_prefix", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "string token",
           tags=frozenset({"task_sensitive", "runtime_sensitive"}), runtime_sensitive=True),
        # --- Python naming, token-derived (6.8) ---
        _d("code.python.naming.identifier_occurrence_length", ANALYZER_CODE_PYTHON_TOKENS,
           "ordered_histogram", HIST, "W1", "identifier occurrence", top_code=65,
           tags=frozenset({"refactoring_sensitive", "api_sensitive"})),
        _d("code.python.naming.identifier_occurrence_case_style", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "identifier occurrence",
           tags=frozenset({"refactoring_sensitive", "api_sensitive"})),
        # --- Python naming, AST-derived (6.8) ---
        _d("code.python.naming.binding_role", ANALYZER_CODE_PYTHON_AST, "categorical_distribution",
           CAT, "JSD2", "binding event",
           tags=frozenset({"parser_dependent", "task_sensitive"})),
        _d("code.python.naming.binding_length", ANALYZER_CODE_PYTHON_AST, "ordered_histogram",
           HIST, "W1", "binding event", top_code=65,
           tags=frozenset({"parser_dependent", "refactoring_sensitive"})),
        _d("code.python.naming.binding_case_style", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "binding event",
           tags=frozenset({"parser_dependent", "refactoring_sensitive"})),
        _d("code.python.naming.binding_component_length", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "binding event", top_code=33,
           tags=frozenset({"parser_dependent", "refactoring_sensitive"})),
        _d("code.python.naming.attribute_name_length", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "attribute occurrence", top_code=65,
           tags=frozenset({"parser_dependent", "api_sensitive", "framework_sensitive"})),
        _d("code.python.naming.attribute_case_style", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "attribute occurrence",
           tags=frozenset({"parser_dependent", "api_sensitive", "framework_sensitive"})),
        # --- Python syntax (6.9) ---
        _d("code.python.syntax.node_distribution", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "ast node",
           tags=frozenset({"parser_dependent", "runtime_sensitive", "task_sensitive",
                           "structural"}), runtime_sensitive=True),
        _d("code.python.syntax.parent_child_distribution", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "ast node edge",
           tags=frozenset({"parser_dependent", "runtime_sensitive", "task_sensitive",
                           "structural"}), runtime_sensitive=True),
        _d("code.python.syntax.node_depth", ANALYZER_CODE_PYTHON_AST, "ordered_histogram", HIST,
           "W1", "ast node", top_code=26,
           tags=frozenset({"parser_dependent", "runtime_sensitive", "task_sensitive",
                           "structural"}), runtime_sensitive=True),
        # --- Python structure (6.10) ---
        _d("code.python.structure.function_count", ANALYZER_CODE_PYTHON_AST, "integer", SUM,
           "NONE", "artifact",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.class_count", ANALYZER_CODE_PYTHON_AST, "integer", SUM, "NONE",
           "artifact", tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.function_kind", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "function",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.function_length_lines", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "function", top_code=201,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural",
                           "framework_sensitive"})),
        _d("code.python.structure.parameter_count", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "function", top_code=21,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.return_count", ANALYZER_CODE_PYTHON_AST, "ordered_histogram",
           HIST, "W1", "function", top_code=11,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.nonterminal_return_function_share", ANALYZER_CODE_PYTHON_AST,
           "ratio", RATIO, "ABS", "function",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.control_construct", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "control construct",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.max_control_nesting", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "function", top_code=11,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.branch_construct_count", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "function", top_code=21,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.decorator_count", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "decorated definition", top_code=11,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural",
                           "framework_sensitive"})),
        _d("code.python.structure.assignment_kind", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "assignment",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.import_kind", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "import",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.import_alias_share", ANALYZER_CODE_PYTHON_AST, "ratio", RATIO,
           "ABS", "import alias",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.exception_construct", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "exception construct",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.comprehension_kind", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "comprehension",
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        _d("code.python.structure.match_case_count", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "match statement", top_code=11,
           tags=frozenset({"parser_dependent", "task_sensitive", "structural"})),
        # --- Python comments/docstrings (6.11) ---
        _d("code.python.comments.comment_count", ANALYZER_CODE_PYTHON_TOKENS, "integer", SUM,
           "NONE", "comment", tags=frozenset({"comment_policy_sensitive"})),
        _d("code.python.comments.comment_line_share", ANALYZER_CODE_PYTHON_TOKENS, "ratio", RATIO,
           "ABS", "line", tags=frozenset({"comment_policy_sensitive"})),
        _d("code.python.comments.comment_length", ANALYZER_CODE_PYTHON_TOKENS,
           "ordered_histogram", HIST, "W1", "comment", top_code=121,
           tags=frozenset({"comment_policy_sensitive"})),
        _d("code.python.comments.docstring_kind", ANALYZER_CODE_PYTHON_AST,
           "categorical_distribution", CAT, "JSD2", "docstring",
           tags=frozenset({"comment_policy_sensitive", "documentation_policy_sensitive"})),
        _d("code.python.comments.docstring_length", ANALYZER_CODE_PYTHON_AST,
           "ordered_histogram", HIST, "W1", "docstring", top_code=501,
           tags=frozenset({"comment_policy_sensitive", "documentation_policy_sensitive"})),
        # --- toolchain markers (6.12) ---
        _d("code.toolchain.python_comment_marker", ANALYZER_CODE_PYTHON_TOKENS,
           "categorical_distribution", CAT, "JSD2", "marker event",
           tags=frozenset({"toolchain_sensitive", "generated_code_sensitive"})),
        # --- tree-sitter parser-backed (6.13) ---
        _d("code.parser.named_node_type", ANALYZER_CODE_TREE_SITTER, "categorical_distribution",
           CAT, "JSD2", "named node",
           tags=frozenset({"parser_dependent", "language_specific", "structural"})),
        _d("code.parser.named_parent_child", ANALYZER_CODE_TREE_SITTER,
           "categorical_distribution", CAT, "JSD2", "named-node edge",
           tags=frozenset({"parser_dependent", "language_specific", "structural"})),
        _d("code.parser.named_depth", ANALYZER_CODE_TREE_SITTER, "ordered_histogram", HIST, "W1",
           "named node", top_code=32,
           tags=frozenset({"parser_dependent", "language_specific", "structural"})),
        _d("code.parser.identifier_length", ANALYZER_CODE_TREE_SITTER, "ordered_histogram", HIST,
           "W1", "identifier occurrence", top_code=64,
           tags=frozenset({"parser_dependent", "language_specific", "content_reduced"})),
        _d("code.parser.identifier_style", ANALYZER_CODE_TREE_SITTER, "categorical_distribution",
           CAT, "JSD2", "identifier occurrence",
           tags=frozenset({"parser_dependent", "language_specific", "content_reduced"})),
        _d("code.parser.comment_kind", ANALYZER_CODE_TREE_SITTER, "categorical_distribution", CAT,
           "JSD2", "comment",
           tags=frozenset({"parser_dependent", "comment_policy_sensitive", "language_specific"})),
        _d("code.parser.comment_length", ANALYZER_CODE_TREE_SITTER, "ordered_histogram", HIST,
           "W1", "comment", top_code=256,
           tags=frozenset({"parser_dependent", "comment_policy_sensitive", "language_specific"})),
        # --- optional linguistic (6.14) ---
        _d("text.linguistic.upos", ANALYZER_TEXT_LINGUISTIC, "categorical_distribution", CAT,
           "JSD2", "linguistic token",
           tags=frozenset({"resource_sensitive", "language_specific", "structural"})),
        _d("text.linguistic.dependency_relation", ANALYZER_TEXT_LINGUISTIC,
           "categorical_distribution", CAT, "JSD2", "linguistic token",
           tags=frozenset({"resource_sensitive", "language_specific", "structural"})),
        _d("text.linguistic.dependency_distance", ANALYZER_TEXT_LINGUISTIC, "ordered_histogram",
           HIST, "W1", "non-root dependency arc", top_code=32,
           tags=frozenset({"resource_sensitive", "language_specific", "structural"})),
        _d("text.linguistic.morph_attribute", ANALYZER_TEXT_LINGUISTIC,
           "categorical_distribution", CAT, "JSD2", "morphology attribute event",
           tags=frozenset({"resource_sensitive", "language_specific", "structural"})),
        _d("text.linguistic.morph_coverage", ANALYZER_TEXT_LINGUISTIC, "ratio", RATIO, "ABS",
           "linguistic token", tags=frozenset({"resource_sensitive", "language_specific"})),
    ]
}

assert len(FEATURES) == len({f.feature_id for f in FEATURES.values()})


def get_feature(feature_id: str) -> FeatureDef:
    return FEATURES[feature_id]


def features_owned_by(analyzer_id: str) -> tuple[FeatureDef, ...]:
    owned = [fdef for fdef in FEATURES.values() if fdef.analyzer_id == analyzer_id]
    return tuple(sorted(owned, key=lambda fdef: fdef.feature_id))
