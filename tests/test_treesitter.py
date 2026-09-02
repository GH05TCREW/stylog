"""Tree-sitter parser adapter and analyzer tests (spec 6.13).

Covers: checked mapping/manifest loading, parse facts error contract, the
seven code.parser.* features per language, hand-verified invariants on tiny
samples, malformed input (parser_error), unsupported grammar (unavailable),
and config disablement (disabled).
"""

from __future__ import annotations

import hashlib

import pytest

from stylog.analysis.registry import (
    ANALYZER_CODE_TREE_SITTER,
    get_feature,
)
from stylog.analysis.treesitter import (
    DIAGNOSTIC_PARSE_ERROR,
    DIAGNOSTIC_UNAVAILABLE,
    TreeSitterAnalyzer,
)
from stylog.config import StylogConfig, parse_config_dict
from stylog.domain.artifact import ArtifactKind
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.feature import FeatureStatus
from stylog.domain.provenance import current_runtime_signature
from stylog.parsers.tree_sitter import (
    SUPPORTED_LANGUAGES,
    TREE_SITTER_RUNTIME_VERSION,
    load_manifest,
    load_manifest_sha256,
    load_mappings,
    parse_tree_sitter,
)
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact

EXPECTED_FEATURES = {
    "code.parser.comment_kind",
    "code.parser.comment_length",
    "code.parser.identifier_length",
    "code.parser.identifier_style",
    "code.parser.named_depth",
    "code.parser.named_node_type",
    "code.parser.named_parent_child",
}

# Representative samples: one line comment, one block comment, identifiers,
# a function definition, and a struct/interface item per language.
SAMPLES = {
    "javascript": (
        "// c1\n/* c2 */\nconst myVar = 1;\n"
        "function myFunc(paramOne) { return paramOne + myVar; }\n"
        "class Foo { bar() {} }\n"
    ),
    "typescript": (
        "// c1\n/* c2 */\nconst myVar: number = 1;\n"
        "function myFunc(paramOne: number): number { return paramOne; }\n"
        "interface Foo { bar: string; }\n"
    ),
    "c": (
        "// c1\n/* c2 */\nint my_var = 1;\nstruct Foo { int field_one; };\n"
        "int my_func(int param_one) { return param_one + my_var; }\n"
    ),
    "rust": (
        "// c1\n/* c2 */\nfn my_func(param_one: i32) -> i32 { param_one + 1 }\n"
        "struct Foo { field_one: i32 }\n"
    ),
}

ROOT_TYPES = {
    "javascript": "program",
    "typescript": "program",
    "c": "translation_unit",
    "rust": "source_file",
}

# Hand-verified identifier-style populations for SAMPLES (identifier nodes only;
# type_identifier/field_identifier/property_identifier are not mapped; the
# javascript class name Foo is a plain identifier node -> pascal).
EXPECTED_STYLES = {
    "javascript": {"camel_lower": 5, "pascal": 1},  # + class name Foo
    "typescript": {"camel_lower": 4},  # myVar, myFunc, paramOne x2
    "c": {"snake_lower": 5},  # my_var, my_func, param_one x2, my_var
    "rust": {"snake_lower": 3},  # my_func, param_one x2
}

EXPECTED_IDENTIFIER_LENGTHS = {
    "javascript": {3: 1, 5: 2, 6: 1, 8: 2},
    "typescript": {5: 1, 6: 1, 8: 2},
    "c": {6: 2, 7: 1, 9: 2},
    "rust": {7: 1, 9: 2},
}


def make_ctx(config: StylogConfig | None = None) -> AnalysisContext:
    return AnalysisContext(
        config=StylogConfig() if config is None else config,
        runtime=current_runtime_signature(),
        resources=ResourceHandles(
            tree_sitter_mappings=load_mappings(),
            grammar_manifest=load_manifest(),
            grammar_manifest_sha256=load_manifest_sha256(),
        ),
    )


def make_artifact(language: str, source: str) -> RuntimeArtifact:
    raw = source.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/000001",
        kind=ArtifactKind.CODE,
        language=language,
        encoding="utf-8",
        raw_bytes=raw,
        text=source,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def run_analyzer(language, source, ctx=None):
    ctx = make_ctx() if ctx is None else ctx
    artifact = make_artifact(language, source)
    facts = parse_tree_sitter(artifact, ctx)
    analyzer = TreeSitterAnalyzer()
    analyzer._language = language
    output = analyzer.analyze(artifact, ctx, facts)
    return artifact, facts, output


def by_id(output):
    return {obs.feature_id: obs for obs in output.observations}


def categorical(obs) -> dict[str, int]:
    return {entry.key: entry.count for entry in obs.value.counts}


def histogram(obs) -> dict[int, int]:
    return {entry.point: entry.count for entry in obs.value.points}


def test_supported_languages_constant() -> None:
    assert set(SUPPORTED_LANGUAGES) == {"javascript", "typescript", "c", "rust"}


def test_owned_features_exactly_seven() -> None:
    analyzer = TreeSitterAnalyzer()
    assert analyzer.analyzer_id == ANALYZER_CODE_TREE_SITTER
    assert analyzer.needs == "tree_sitter_parse"
    assert set(analyzer.owned_feature_ids()) == EXPECTED_FEATURES


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_parse_and_analyze_ok_per_language(language: str) -> None:
    _, facts, output = run_analyzer(language, SAMPLES[language])
    assert facts.error_code is None
    assert facts.root is not None
    assert facts.language == language

    # Exactly one observation per owned feature, sorted by feature_id.
    feature_ids = [obs.feature_id for obs in output.observations]
    assert set(feature_ids) == EXPECTED_FEATURES
    assert feature_ids == sorted(feature_ids)
    assert output.diagnostics == ()

    obs = by_id(output)
    assert all(o.status == FeatureStatus.OK for o in obs.values())
    for feature_id, o in obs.items():
        assert o.support.kind == get_feature(feature_id).support_kind

    # Root invariants: named root counted once at depth 0.
    assert categorical(obs["code.parser.named_node_type"])[ROOT_TYPES[language]] == 1
    assert histogram(obs["code.parser.named_depth"])[0] == 1
    # Edges: every named node except the root contributes exactly one edge.
    assert obs["code.parser.named_parent_child"].value.total == (
        obs["code.parser.named_node_type"].value.total - 1
    )
    assert obs["code.parser.named_parent_child"].support.count == (
        obs["code.parser.named_parent_child"].value.total
    )

    # Comments: "// c1" (5 code points) line, "/* c2 */" (8 code points) block.
    assert categorical(obs["code.parser.comment_kind"]) == {"line": 1, "block": 1}
    assert histogram(obs["code.parser.comment_length"]) == {5: 1, 8: 1}
    assert obs["code.parser.comment_length"].support.count == 2

    # Identifiers per checked mapping (only "identifier" node type).
    assert categorical(obs["code.parser.identifier_style"]) == EXPECTED_STYLES[language]
    assert histogram(obs["code.parser.identifier_length"]) == (
        EXPECTED_IDENTIFIER_LENGTHS[language]
    )


def test_javascript_tiny_sample_invariants() -> None:
    # Hand-computed: program > lexical_declaration > variable_declarator > (identifier, number)
    _, facts, output = run_analyzer("javascript", "const x = 1;\n")
    assert facts.error_code is None
    obs = by_id(output)

    node_types = categorical(obs["code.parser.named_node_type"])
    assert node_types == {
        "program": 1,
        "lexical_declaration": 1,
        "variable_declarator": 1,
        "identifier": 1,
        "number": 1,
    }
    assert obs["code.parser.named_node_type"].support.count == 5

    edges = categorical(obs["code.parser.named_parent_child"])
    assert edges == {
        "program→lexical_declaration": 1,
        "lexical_declaration→variable_declarator": 1,
        "variable_declarator→identifier": 1,
        "variable_declarator→number": 1,
    }

    assert histogram(obs["code.parser.named_depth"]) == {0: 1, 1: 1, 2: 1, 3: 2}
    assert obs["code.parser.named_depth"].value.top_code == 32

    assert histogram(obs["code.parser.identifier_length"]) == {1: 1}
    assert obs["code.parser.identifier_length"].value.total == 1
    assert categorical(obs["code.parser.identifier_style"]) == {"lower": 1}

    # No comments in this sample -> typed insufficient_support, not empty/zero.
    assert obs["code.parser.comment_kind"].status == FeatureStatus.INSUFFICIENT_SUPPORT
    assert obs["code.parser.comment_length"].status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_malformed_javascript_parser_error() -> None:
    artifact, facts, output = run_analyzer("javascript", "const = ;")
    assert facts.error_code == "PARSER_ERROR"
    assert facts.root is None
    assert len(output.observations) == 7
    assert all(
        obs.status == FeatureStatus.PARSER_ERROR for obs in output.observations
    )
    assert len(output.diagnostics) == 1
    diagnostic = output.diagnostics[0]
    assert diagnostic.code == DIAGNOSTIC_PARSE_ERROR
    assert diagnostic.severity == DiagnosticSeverity.ERROR
    assert diagnostic.analyzer_id == ANALYZER_CODE_TREE_SITTER
    assert diagnostic.artifact_id == artifact.artifact_id
    # Generic code-surface features are owned by another analyzer and are
    # unaffected by this parser failure (not exercised here).


def test_unsupported_language_unavailable() -> None:
    # parse_tree_sitter is only dispatched for the four grammar languages;
    # anything else degrades deterministically to UNAVAILABLE.
    _, facts, output = run_analyzer("python", "x = 1\n")
    assert facts.error_code == "UNAVAILABLE"
    assert facts.root is None
    assert all(
        obs.status == FeatureStatus.UNAVAILABLE for obs in output.observations
    )
    assert len(output.diagnostics) == 1
    assert output.diagnostics[0].code == DIAGNOSTIC_UNAVAILABLE
    assert output.diagnostics[0].severity == DiagnosticSeverity.ERROR


def test_disabled_by_config() -> None:
    config = parse_config_dict({"analysis": {"code": {"tree_sitter": {"enabled": False}}}})
    _, facts, output = run_analyzer("c", SAMPLES["c"], make_ctx(config))
    assert facts.error_code is None  # parsing still succeeds
    assert len(output.observations) == 7
    assert all(obs.status == FeatureStatus.DISABLED for obs in output.observations)
    assert output.diagnostics == ()


def test_language_must_be_set_by_engine() -> None:
    ctx = make_ctx()
    artifact = make_artifact("javascript", "const x = 1;\n")
    facts = parse_tree_sitter(artifact, ctx)
    analyzer = TreeSitterAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.signature(ctx)
    with pytest.raises(RuntimeError):
        analyzer.resources(ctx)
    with pytest.raises(RuntimeError):
        analyzer.analyze(artifact, ctx, facts)


def test_signature_is_per_language() -> None:
    ctx = make_ctx()
    manifest = ctx.resources.grammar_manifest
    for language in sorted(SAMPLES):
        analyzer = TreeSitterAnalyzer()
        analyzer._language = language
        signature = analyzer.signature(ctx)
        entry = manifest[language]

        assert signature.analyzer_id == ANALYZER_CODE_TREE_SITTER
        assert [r.id for r in signature.resources] == [
            f"stylog.tree_sitter.mapping.{language}"
        ]
        mapping = ctx.resources.tree_sitter_mappings[language]
        resource = signature.resources[0]
        assert resource.version == mapping.version
        assert resource.sha256 == mapping.sha256

        backend = signature.backend
        assert backend.backend_id == "tree-sitter"
        assert backend.implementation_version == TREE_SITTER_RUNTIME_VERSION
        assert backend.scientific_compatibility_id == f"stylog.tree-sitter.{language}/1"
        assert [(p.package, p.version) for p in backend.packages] == [
            ("tree-sitter", backend.implementation_version),
            (entry.package, entry.installed_version),
        ]
        grammar = backend.parser_grammar
        assert grammar is not None
        assert grammar.language == language
        assert grammar.grammar_id == entry.grammar_id
        assert grammar.grammar_version == entry.installed_version
        assert grammar.grammar_revision == entry.upstream_revision
        assert grammar.node_types_sha256 == entry.node_types_sha256
        assert grammar.grammar_manifest_sha256 == load_manifest_sha256()
        assert grammar.language_abi_version == entry.abi_version
