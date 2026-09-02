"""Tree-sitter parser-backed analyzer (spec 6.13).

Owns the seven ``code.parser.*`` features for javascript/typescript/c/rust.
All measurement is a pure function of parse facts, the checked per-language
mapping resource, and the raw bytes; unmapped node types are never guessed.

Diagnostic codes (analyzer-specific, severity error; the mandatory list in
spec 10.4 has no tree-sitter code): ``TREE_SITTER_UNAVAILABLE`` when the
grammar/runtime prerequisite is missing, ``TREE_SITTER_PARSE_ERROR`` when the
grammar rejected the input.
"""

from __future__ import annotations

from typing import Any

from stylog.analysis import build
from stylog.analysis.base import AnalyzerOutput, BaseAnalyzer, all_status_observations
from stylog.analysis.identifiers import classify_style
from stylog.analysis.registry import (
    ANALYZER_CODE_TREE_SITTER,
    FEATURE_REGISTRY_VERSION,
    get_feature,
)
from stylog.domain.diagnostic import Diagnostic, DiagnosticSeverity
from stylog.domain.feature import FeatureObservation, FeatureStatus
from stylog.domain.provenance import AnalyzerSignature, ResourceSignature
from stylog.parsers.tree_sitter import TreeSitterParseFacts, tree_sitter_backend_signature
from stylog.runtime import AnalysisContext, RuntimeArtifact, TreeSitterLanguageMapping

FEATURE_COMMENT_KIND = "code.parser.comment_kind"
FEATURE_COMMENT_LENGTH = "code.parser.comment_length"
FEATURE_IDENTIFIER_LENGTH = "code.parser.identifier_length"
FEATURE_IDENTIFIER_STYLE = "code.parser.identifier_style"
FEATURE_NAMED_DEPTH = "code.parser.named_depth"
FEATURE_NAMED_NODE_TYPE = "code.parser.named_node_type"
FEATURE_NAMED_PARENT_CHILD = "code.parser.named_parent_child"

DIAGNOSTIC_UNAVAILABLE = "TREE_SITTER_UNAVAILABLE"
DIAGNOSTIC_PARSE_ERROR = "TREE_SITTER_PARSE_ERROR"

_PARENT_CHILD_SEPARATOR = "\u2192"  # U+2192, per spec 6.13 parent-child key


class TreeSitterAnalyzer(BaseAnalyzer):
    """Parser-backed analyzer; the engine sets ``_language`` per artifact."""

    analyzer_id = ANALYZER_CODE_TREE_SITTER
    needs = "tree_sitter_parse"

    def __init__(self) -> None:
        self._language: str | None = None

    def _require_language(self) -> str:
        if self._language is None:
            raise RuntimeError(
                "TreeSitterAnalyzer._language must be set by the engine "
                "before signature/resources/analyze"
            )
        return self._language

    def _mapping(self, ctx: AnalysisContext) -> TreeSitterLanguageMapping:
        return ctx.resources.tree_sitter_mappings[self._require_language()]

    def resources(self, ctx: AnalysisContext) -> tuple[ResourceSignature, ...]:
        mapping = self._mapping(ctx)
        return (
            ResourceSignature(
                id=f"stylog.tree_sitter.mapping.{mapping.language}",
                version=mapping.version,
                sha256=mapping.sha256,
            ),
        )

    def signature(self, ctx: AnalysisContext) -> AnalyzerSignature:
        language = self._require_language()
        entry = ctx.resources.grammar_manifest[language]
        return AnalyzerSignature(
            analyzer_id=self.analyzer_id,
            implementation_version=self.implementation_version,
            feature_registry_version=FEATURE_REGISTRY_VERSION,
            backend=tree_sitter_backend_signature(language, entry),
            resources=tuple(sorted(self.resources(ctx), key=lambda sig: sig.id)),
        )

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        self._require_language()  # engine must have set the artifact language
        if not isinstance(facts, TreeSitterParseFacts):
            raise RuntimeError(
                "TreeSitterAnalyzer requires TreeSitterParseFacts from the engine"
            )
        fdefs = self.owned_feature_ids()
        if facts.error_code is not None:
            observation_status = (
                FeatureStatus.UNAVAILABLE
                if facts.error_code == "UNAVAILABLE"
                else FeatureStatus.PARSER_ERROR
            )
            diagnostic_code = (
                DIAGNOSTIC_UNAVAILABLE
                if facts.error_code == "UNAVAILABLE"
                else DIAGNOSTIC_PARSE_ERROR
            )
            diagnostic = Diagnostic(
                code=diagnostic_code,
                severity=DiagnosticSeverity.ERROR,
                analyzer_id=self.analyzer_id,
                artifact_id=artifact.artifact_id,
            )
            return AnalyzerOutput(
                observations=self._all_status(observation_status),
                diagnostics=(diagnostic,),
            )
        if not ctx.config.analysis.code.tree_sitter.enabled:
            return AnalyzerOutput(
                observations=self._all_status(FeatureStatus.DISABLED)
            )
        return AnalyzerOutput(
            observations=self._measure(fdefs, artifact, self._mapping(ctx), facts)
        )

    def _all_status(self, status: FeatureStatus) -> tuple[FeatureObservation, ...]:
        return all_status_observations(self.analyzer_id, self.implementation_version, status)

    def _measure(
        self,
        feature_ids: tuple[str, ...],
        artifact: RuntimeArtifact,
        mapping: TreeSitterLanguageMapping,
        facts: TreeSitterParseFacts,
    ) -> tuple[FeatureObservation, ...]:
        raw = artifact.raw_bytes
        node_type_counts: dict[str, int] = {}
        edge_counts: dict[str, int] = {}
        depth_values: list[int] = []
        identifier_lengths: list[int] = []
        style_counts: dict[str, int] = {}
        comment_kind_counts: dict[str, int] = {}
        comment_lengths: list[int] = []

        # Iterative walk with explicit stack; each entry carries the named
        # ancestry context: (node, named ancestor count, nearest named ancestor type).
        stack: list[tuple[Any, int, str | None]] = [(facts.root, 0, None)]
        while stack:
            node, named_ancestors, nearest_named = stack.pop()
            if node.is_named:
                node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
                depth_values.append(named_ancestors)
                if nearest_named is not None:
                    key = nearest_named + _PARENT_CHILD_SEPARATOR + node.type
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                if node.type in mapping.identifier_node_types:
                    text = raw[node.start_byte : node.end_byte].decode("utf-8")
                    identifier_lengths.append(len(text))
                    style = classify_style(text)
                    if style != "discard":  # exactly "_" per spec 8.9
                        style_counts[style] = style_counts.get(style, 0) + 1
                if node.type in mapping.comment_node_types:
                    text = raw[node.start_byte : node.end_byte].decode("utf-8")
                    if text.startswith(mapping.line_comment_delimiters):
                        kind = "line"
                    elif text.startswith(mapping.block_comment_delimiters):
                        kind = "block"
                    else:
                        raise ValueError(
                            f"comment node text matches no checked delimiter: {text[:16]!r}"
                        )
                    comment_kind_counts[kind] = comment_kind_counts.get(kind, 0) + 1
                    comment_lengths.append(len(text))
            child_named_ancestors = named_ancestors + 1 if node.is_named else named_ancestors
            child_nearest_named = node.type if node.is_named else nearest_named
            for child in reversed(node.children):
                stack.append((child, child_named_ancestors, child_nearest_named))

        def hist(feature_id: str, values: list[int]) -> Any:
            top_code = get_feature(feature_id).top_code
            assert top_code is not None
            return build.histogram_value(values, top_code)

        results: dict[str, tuple[Any, int]] = {
            FEATURE_COMMENT_KIND: (
                build.categorical_value(comment_kind_counts),
                sum(comment_kind_counts.values()),
            ),
            FEATURE_COMMENT_LENGTH: (
                hist(FEATURE_COMMENT_LENGTH, comment_lengths),
                len(comment_lengths),
            ),
            FEATURE_IDENTIFIER_LENGTH: (
                hist(FEATURE_IDENTIFIER_LENGTH, identifier_lengths),
                len(identifier_lengths),
            ),
            FEATURE_IDENTIFIER_STYLE: (
                build.categorical_value(style_counts),
                sum(style_counts.values()),
            ),
            FEATURE_NAMED_DEPTH: (
                hist(FEATURE_NAMED_DEPTH, depth_values),
                len(depth_values),
            ),
            FEATURE_NAMED_NODE_TYPE: (
                build.categorical_value(node_type_counts),
                sum(node_type_counts.values()),
            ),
            FEATURE_NAMED_PARENT_CHILD: (
                build.categorical_value(edge_counts),
                sum(edge_counts.values()),
            ),
        }
        assert set(results) == set(feature_ids)

        observations: list[FeatureObservation] = []
        for feature_id in feature_ids:
            fdef = get_feature(feature_id)
            value, support_count = results[feature_id]
            observations.append(
                build.value_observation(
                    fdef, self.analyzer_id, self.implementation_version, value, support_count
                )
            )
        return tuple(observations)
