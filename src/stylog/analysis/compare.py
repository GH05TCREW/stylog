"""Comparison mathematics (spec section 12).

A Comparison is an ordered set of independently interpretable per-feature
components grouped by family. There is deliberately no global similarity
score (12.1). Missing or gated-out features are omitted from components and
represented in diagnostics, never as maximal or zero distance (12.9).
"""

from __future__ import annotations

from stylog.analysis import compat, stats
from stylog.analysis.registry import FEATURES, FeatureDef
from stylog.domain.diagnostic import (
    Diagnostic,
    DiagnosticSeverity,
    make_diagnostic,
    sort_diagnostics,
)
from stylog.domain.evidence import AggregateObservation, AggregationKind, EvidenceAggregate
from stylog.domain.feature import (
    CategoricalDistributionValue,
    FeatureObservation,
    FeatureValue,
    OkFeatureObservation,
    OrderedHistogramValue,
    RatioValue,
    Support,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Comparison, ComparisonComponent, ComparisonFamily
from stylog.exceptions import PortableArtifactError

NO_COMPARABLE_FEATURES = "NO_COMPARABLE_FEATURES"
FEATURE_NOT_COMPARABLE = "FEATURE_NOT_COMPARABLE"

_MISSING_STATUS = "missing"
_OK_STATUS = "ok"

_ABS_RATIO_UNIT = "proportion points on [0,1]"
_ABS_UNIT = "absolute difference"
_SPD_UNIT = "symmetric proportional distance [0,2]"
_JSD2_UNIT = "jensen-shannon distance [0,1]"
_SAMPLE_W1_METRIC = "sample_wasserstein_1"
_SAMPLE_W1_UNIT = "sample units"

# Native transformed unit per W1 feature (spec 12.6).
_W1_UNITS: dict[str, str] = {
    "text.lexical.word_length": "word code points",
    "text.structure.sentence_length_tokens": "sentence tokens",
    "text.structure.sentence_length_characters": "sentence code points",
    "text.structure.paragraph_sentence_count": "paragraph sentences",
    "text.structure.paragraph_token_count": "paragraph tokens",
    "code.surface.indent_char_count": "indent characters",
    "code.surface.nonblank_line_length": "line code points",
    "code.surface.blank_run_length": "blank lines",
    "code.python.naming.identifier_occurrence_length": "code points",
    "code.python.naming.binding_length": "code points",
    "code.python.naming.binding_component_length": "code points",
    "code.python.naming.attribute_name_length": "code points",
    "code.python.syntax.node_depth": "AST depth",
    "code.python.structure.function_length_lines": "lines",
    "code.python.structure.parameter_count": "parameters",
    "code.python.structure.return_count": "returns",
    "code.python.structure.max_control_nesting": "nesting levels",
    "code.python.structure.branch_construct_count": "branch constructs",
    "code.python.structure.decorator_count": "decorators",
    "code.python.structure.match_case_count": "match cases",
    "code.python.comments.comment_length": "comment code points",
    "code.python.comments.docstring_length": "docstring code points",
    "code.parser.named_depth": "named-node depth",
    "code.parser.identifier_length": "code points",
    "code.parser.comment_length": "comment code points",
    "text.linguistic.dependency_distance": "token distance",
}
_W1_DEFAULT_UNIT = "top-coded units"


def _comparison_omission_diagnostic(
    feature_id: str,
    left_status: str,
    right_status: str,
) -> Diagnostic:
    """Describe why a registry-comparable feature emitted no component."""
    return make_diagnostic(
        FEATURE_NOT_COMPARABLE,
        DiagnosticSeverity.WARNING,
        feature_id=feature_id,
        context=(("left_status", left_status), ("right_status", right_status)),
    )


def _observation_status(observation: FeatureObservation | None) -> str:
    return _MISSING_STATUS if observation is None else str(observation.status)


def _aggregate_status(
    aggregate: AggregateObservation | None,
    reducer: AggregationKind,
) -> str:
    """Return the comparison status of one aggregate feature value."""
    if aggregate is None:
        return _MISSING_STATUS
    value = (
        aggregate.sample_values
        if reducer is AggregationKind.SAMPLE_SUMMARY
        else aggregate.pooled
    )
    if value is not None:
        return _OK_STATUS
    if aggregate.contributing_samples == 0 and len(aggregate.missing) == 1:
        return str(aggregate.missing[0].status)
    return _MISSING_STATUS


def _compare_values(
    fdef: FeatureDef,
    left_value: FeatureValue,
    right_value: FeatureValue,
    left_support: Support,
    right_support: Support,
    semantic_version: str,
) -> ComparisonComponent | None:
    """One comparison component under the feature's registry metric."""
    metric = fdef.metric
    if metric == "ABS":
        left_scalar = compat.primary_scalar(left_value)
        right_scalar = compat.primary_scalar(right_value)
        if left_scalar is None or right_scalar is None:
            return None
        value = stats.abs_distance(left_scalar, right_scalar)
        unit = _ABS_RATIO_UNIT if isinstance(left_value, RatioValue) else _ABS_UNIT
    elif metric == "SPD":
        left_scalar = compat.primary_scalar(left_value)
        right_scalar = compat.primary_scalar(right_value)
        if left_scalar is None or right_scalar is None:
            return None
        value = stats.symmetric_proportional_distance(left_scalar, right_scalar)
        unit = _SPD_UNIT
    elif metric == "JSD2":
        if not isinstance(left_value, CategoricalDistributionValue) or not isinstance(
            right_value, CategoricalDistributionValue
        ):
            return None
        value = stats.jensen_shannon_distance2(
            {entry.key: entry.count for entry in left_value.counts},
            left_value.total,
            {entry.key: entry.count for entry in right_value.counts},
            right_value.total,
        )
        unit = _JSD2_UNIT
    elif metric == "W1":
        if not isinstance(left_value, OrderedHistogramValue) or not isinstance(
            right_value, OrderedHistogramValue
        ):
            return None
        if left_value.top_code != right_value.top_code:
            return None
        value = stats.wasserstein_1(
            {entry.point: entry.count for entry in left_value.points},
            left_value.total,
            {entry.point: entry.count for entry in right_value.points},
            right_value.total,
        )
        unit = _W1_UNITS.get(fdef.feature_id, _W1_DEFAULT_UNIT)
    else:
        return None
    return ComparisonComponent(
        feature_id=fdef.feature_id,
        semantic_version=semantic_version,
        metric=metric,
        value=value,
        unit=unit,
        left_support=left_support,
        right_support=right_support,
    )


def _finish(
    left_ref: str,
    right_ref: str,
    components_by_family: dict[str, list[ComparisonComponent]],
    diagnostics: list[Diagnostic],
) -> Comparison:
    families = tuple(
        ComparisonFamily(family=family, components=tuple(components_by_family[family]))
        for family in sorted(components_by_family)
    )
    if not families:
        diagnostics.append(
            make_diagnostic(NO_COMPARABLE_FEATURES, DiagnosticSeverity.WARNING)
        )
    return Comparison(
        left_ref=left_ref,
        right_ref=right_ref,
        families=families,
        diagnostics=sort_diagnostics(diagnostics),
    )


def compare_fingerprints(
    left: Fingerprint,
    right: Fingerprint,
    left_ref: str,
    right_ref: str,
) -> Comparison:
    """Compare two fingerprints feature-by-feature (spec 12.2-12.9)."""
    if left.artifact.kind != right.artifact.kind:
        raise PortableArtifactError(
            "text and code primary artifacts must not be cross-compared "
            f"(left kind {left.artifact.kind!r}, right kind {right.artifact.kind!r})"
        )
    universe = sorted(
        feature_id
        for feature_id in {
            observation.feature_id for observation in left.features
        } | {observation.feature_id for observation in right.features}
        if feature_id in FEATURES and FEATURES[feature_id].metric != "NONE"
    )
    diagnostics: list[Diagnostic] = []
    components_by_family: dict[str, list[ComparisonComponent]] = {}
    for feature_id in universe:
        fdef = FEATURES[feature_id]
        left_observation = compat.observation_for(left, feature_id)
        right_observation = compat.observation_for(right, feature_id)
        if not isinstance(left_observation, OkFeatureObservation) or not isinstance(
            right_observation, OkFeatureObservation
        ):
            diagnostics.append(
                _comparison_omission_diagnostic(
                    feature_id,
                    _observation_status(left_observation),
                    _observation_status(right_observation),
                )
            )
            continue
        mismatch = compat.observation_pair_mismatch(fdef, left, right)
        if mismatch is not None:
            diagnostics.append(
                make_diagnostic(mismatch, DiagnosticSeverity.WARNING, feature_id=feature_id)
            )
            continue
        component = _compare_values(
            fdef,
            left_observation.value,
            right_observation.value,
            left_observation.support,
            right_observation.support,
            left_observation.semantic_version,
        )
        if component is None:
            continue
        components_by_family.setdefault(fdef.family, []).append(component)
    return _finish(left_ref, right_ref, components_by_family, diagnostics)


def compare_aggregates(
    left: EvidenceAggregate,
    right: EvidenceAggregate,
    left_ref: str,
    right_ref: str,
) -> Comparison:
    """Compare two evidence aggregates (spec 12.10).

    Pooled values are compared with the feature's ordinary metric; sample
    summary features compare sample values via Wasserstein-1 on the actual
    scalars under the metric name ``sample_wasserstein_1``.
    """
    left_by_id = {aggregate.feature_id: aggregate for aggregate in left.aggregates}
    right_by_id = {aggregate.feature_id: aggregate for aggregate in right.aggregates}
    universe = sorted(
        feature_id
        for feature_id in left_by_id.keys() | right_by_id.keys()
        if feature_id in FEATURES
        and FEATURES[feature_id].metric != "NONE"
        and FEATURES[feature_id].reducer is not AggregationKind.NOT_AGGREGATABLE
    )
    diagnostics: list[Diagnostic] = []
    components_by_family: dict[str, list[ComparisonComponent]] = {}
    for feature_id in universe:
        fdef = FEATURES[feature_id]
        left_aggregate = left_by_id.get(feature_id)
        right_aggregate = right_by_id.get(feature_id)
        left_status = _aggregate_status(left_aggregate, fdef.reducer)
        right_status = _aggregate_status(right_aggregate, fdef.reducer)
        if left_status != _OK_STATUS or right_status != _OK_STATUS:
            diagnostics.append(
                _comparison_omission_diagnostic(feature_id, left_status, right_status)
            )
            continue
        assert left_aggregate is not None and right_aggregate is not None
        if left_aggregate.semantic_version != right_aggregate.semantic_version:
            diagnostics.append(
                make_diagnostic(
                    compat.FEATURE_SEMANTIC_MISMATCH,
                    DiagnosticSeverity.WARNING,
                    feature_id=feature_id,
                )
            )
            continue
        left_support = Support(kind="sample", count=left_aggregate.contributing_samples)
        right_support = Support(kind="sample", count=right_aggregate.contributing_samples)
        if fdef.reducer is AggregationKind.SAMPLE_SUMMARY:
            assert left_aggregate.sample_values is not None
            assert right_aggregate.sample_values is not None
            component: ComparisonComponent | None = ComparisonComponent(
                feature_id=feature_id,
                semantic_version=left_aggregate.semantic_version,
                metric=_SAMPLE_W1_METRIC,
                value=stats.wasserstein_1_samples(
                    left_aggregate.sample_values, right_aggregate.sample_values
                ),
                unit=_SAMPLE_W1_UNIT,
                left_support=left_support,
                right_support=right_support,
            )
        else:
            assert left_aggregate.pooled is not None
            assert right_aggregate.pooled is not None
            component = _compare_values(
                fdef,
                left_aggregate.pooled,
                right_aggregate.pooled,
                left_support,
                right_support,
                left_aggregate.semantic_version,
            )
        if component is None:
            continue
        components_by_family.setdefault(fdef.family, []).append(component)
    return _finish(left_ref, right_ref, components_by_family, diagnostics)
