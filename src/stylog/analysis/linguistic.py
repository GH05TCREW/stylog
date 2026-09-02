"""spaCy-backed linguistic analyzer (spec 6.14; provenance 5.17/15; local-only 22.8).

Owns the five ``text.linguistic.*`` features. All measurements are pure
functions of the decoded text and the loaded spaCy pipeline; only annotation
label strings (UPOS tags, dependency relation names, morphological
feature/value pairs) are serialized -- never lemma, token, or dependency text.

Local-only loading (spec 22.8): models come exclusively from installed model
packages or existing local directories via ``spacy.load``, which performs no
downloads; ``spacy.cli.download`` is never called. spaCy is imported lazily
inside :func:`load_spacy_model` only (spec 4.14): importing this module never
imports spaCy.

Annotation gating is deliberately conservative (spec 6.14: "Missing annotation
-> typed unavailable, no heuristic substitution"). An annotation family is
emitted only when (a) the pipeline contains a component that can produce it
AND (b) a non-empty processed doc actually carries it (``Doc.has_annotation``).
The doc-level check is skipped for empty docs so that empty input degrades to
``insufficient_support`` (empty population) rather than ``unavailable``. Note
that English spaCy pipelines such as en_core_web_sm populate ``token.morph``
through the ``attribute_ruler`` (tag-to-morph exception mappings), not a
``morphologizer`` component, so the morph producer set includes both. Any
doubt downgrades a feature to typed unavailability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stylog.analysis import build
from stylog.analysis.base import AnalyzerOutput, BaseAnalyzer, all_status_observations
from stylog.analysis.registry import (
    ANALYZER_TEXT_LINGUISTIC,
    FEATURE_REGISTRY_VERSION,
    get_feature,
)
from stylog.capability import require_capability
from stylog.domain.feature import FeatureObservation, FeatureStatus
from stylog.domain.provenance import (
    AnalyzerSignature,
    BackendSignature,
    ModelSignature,
    PackageProvenance,
)
from stylog.exceptions import ResourceError
from stylog.infrastructure.resources import resource_tree_sha256
from stylog.ports import RuntimeServices
from stylog.runtime import AnalysisContext, RuntimeArtifact
from stylog.serialization.canonical import sha256_hex

if TYPE_CHECKING:
    import spacy

FEATURE_UPOS = "text.linguistic.upos"
FEATURE_DEPENDENCY_RELATION = "text.linguistic.dependency_relation"
FEATURE_DEPENDENCY_DISTANCE = "text.linguistic.dependency_distance"
FEATURE_MORPH_ATTRIBUTE = "text.linguistic.morph_attribute"
FEATURE_MORPH_COVERAGE = "text.linguistic.morph_coverage"

SPACY_BACKEND_ID = "spacy"
SPACY_COMPATIBILITY_ID = "stylog.spacy/1"
PREPROCESSING_ID = "stylog.spacy.preprocessing"
PREPROCESSING_VERSION = "1.0.0"

# Pipeline components capable of producing each annotation family. Morphology
# may come from a morphologizer or from attribute-ruler tag mappings (English).
_POS_PRODUCERS = frozenset({"tagger", "morphologizer"})
_DEP_PRODUCERS = frozenset({"parser"})
_MORPH_PRODUCERS = frozenset({"morphologizer", "attribute_ruler"})


def load_spacy_model(
    model_ref: str, services: RuntimeServices
) -> tuple[spacy.Language, BackendSignature]:
    """Load a locally provisioned spaCy model and its backend signature.

    ``model_ref`` is either an existing local directory path or the name of an
    installed model package; anything else raises :class:`ResourceError`
    because the model is not provisioned locally. No network access occurs:
    ``spacy.load`` only resolves installed packages and local paths. The
    ``services`` argument is part of the bootstrap call contract
    (:func:`stylog.bootstrap.build_resource_handles`); the model tree hash is
    taken from ``nlp.path`` directly, so no resolver lookup is needed.
    """
    spacy = require_capability("spacy", "nlp")

    path = Path(model_ref)
    load_ref = str(path) if path.is_dir() else model_ref
    try:
        nlp = spacy.load(load_ref)
    except Exception as exc:
        raise ResourceError(
            f"RESOURCE_MISMATCH: spaCy model not provisioned locally: {model_ref!r}; "
            "install the model package (e.g. pip install <model wheel>) or pass an "
            "existing local model directory before analysis"
        ) from exc

    model_id = str(nlp.meta.get("name") or model_ref)
    model_revision = str(nlp.meta.get("version") or "unknown")
    if nlp.path is not None:
        tree_sha256 = resource_tree_sha256(Path(nlp.path))
    else:
        # No on-disk tree (programmatically built pipeline): hash the canonical
        # meta JSON so the signature remains a stable content identity.
        canonical_meta = json.dumps(nlp.meta, sort_keys=True, separators=(",", ":"))
        tree_sha256 = sha256_hex(canonical_meta.encode("utf-8"))
    model = ModelSignature(
        model_id=model_id,
        model_revision=model_revision,
        model_tree_sha256=tree_sha256,
        # The tokenizer lives inside the model package, so it shares the tree hash.
        tokenizer_id=f"{model_id}/tokenizer",
        tokenizer_version=spacy.__version__,
        tokenizer_tree_sha256=tree_sha256,
        preprocessing_id=PREPROCESSING_ID,
        preprocessing_version=PREPROCESSING_VERSION,
    )
    backend = BackendSignature(
        backend_id=SPACY_BACKEND_ID,
        implementation_version=spacy.__version__,
        scientific_compatibility_id=SPACY_COMPATIBILITY_ID,
        packages=(PackageProvenance(package="spacy", version=spacy.__version__),),
        model=model,
    )
    return nlp, backend


def _has_annotation(
    doc: spacy.tokens.Doc, pipes: frozenset[str], producers: frozenset[str], attr: str
) -> bool:
    """Conservative availability check for one annotation family (module docstring)."""
    if pipes.isdisjoint(producers):
        return False
    if len(doc) == 0:
        return True
    return bool(doc.has_annotation(attr))


def _morph_pairs(token: spacy.tokens.Token) -> tuple[str, ...]:
    """One ``Feature=Value`` category per morphological attribute of ``token``.

    ``MorphAnalysis.to_dict()`` is the normalized mapping: spaCy sorts each
    field's values ascending and comma-joins them, so a multi-valued attribute
    yields exactly one category (e.g. ``PronType=Art,Prs``). Stylog applies no
    further reordering; empty feature or value strings are skipped defensively
    and never become empty-string categories.
    """
    return tuple(
        f"{feature}={value}"
        for feature, value in sorted(token.morph.to_dict().items())
        if feature and value
    )


class LinguisticAnalyzer(BaseAnalyzer):
    """spaCy-backed analyzer for the ``text.linguistic.*`` features (spec 6.14)."""

    analyzer_id = ANALYZER_TEXT_LINGUISTIC
    needs = "none"

    def signature(self, ctx: AnalysisContext) -> AnalyzerSignature:
        return AnalyzerSignature(
            analyzer_id=self.analyzer_id,
            implementation_version=self.implementation_version,
            feature_registry_version=FEATURE_REGISTRY_VERSION,
            backend=self._backend(ctx),
            resources=(),
        )

    @staticmethod
    def _backend(ctx: AnalysisContext) -> BackendSignature:
        backend = ctx.resources.nlp_model_backend
        if backend is None:
            return BackendSignature(
                backend_id=SPACY_BACKEND_ID,
                implementation_version="unknown",
                scientific_compatibility_id=SPACY_COMPATIBILITY_ID,
            )
        return backend

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        feature_ids = self.owned_feature_ids()
        nlp = ctx.resources.nlp_model
        if nlp is None:
            # The engine normally gates this analyzer on model presence; this
            # path is defensive only.
            return AnalyzerOutput(
                observations=all_status_observations(
                    self.analyzer_id, self.implementation_version, FeatureStatus.UNAVAILABLE
                )
            )

        doc = nlp(artifact.text)
        pipes = frozenset(nlp.pipe_names)
        has_pos = _has_annotation(doc, pipes, _POS_PRODUCERS, "POS")
        has_dep = _has_annotation(doc, pipes, _DEP_PRODUCERS, "DEP")
        has_morph = _has_annotation(doc, pipes, _MORPH_PRODUCERS, "MORPH")

        # Linguistic token = spaCy token with token.is_space == False (spec 6.14).
        # Core tokenization (stylog.text.tokenizer) is not reused for these labels.
        linguistic_tokens = [token for token in doc if not token.is_space]

        pos_counts: dict[str, int] = {}
        dep_counts: dict[str, int] = {}
        distances: list[int] = []
        morph_counts: dict[str, int] = {}
        tokens_with_morph = 0

        for token in linguistic_tokens:
            if has_pos:
                pos = token.pos_
                if pos:  # empty labels never become empty-string categories
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
            if has_dep:
                dep = token.dep_
                if dep:
                    dep_counts[dep] = dep_counts.get(dep, 0) + 1
                # Root tokens (token.head == token) contribute no arc.
                if token.head.i != token.i and token.head.doc is doc:
                    distances.append(abs(token.i - token.head.i))
            if has_morph:
                pairs = _morph_pairs(token)
                if pairs:
                    tokens_with_morph += 1
                    for pair in pairs:
                        morph_counts[pair] = morph_counts.get(pair, 0) + 1

        distance_top_code = get_feature(FEATURE_DEPENDENCY_DISTANCE).top_code
        assert distance_top_code is not None
        values: dict[str, tuple[Any, int]] = {
            FEATURE_UPOS: (
                build.categorical_value(pos_counts),
                sum(pos_counts.values()),
            ),
            FEATURE_DEPENDENCY_RELATION: (
                build.categorical_value(dep_counts),
                sum(dep_counts.values()),
            ),
            FEATURE_DEPENDENCY_DISTANCE: (
                build.histogram_value(distances, distance_top_code),
                len(distances),
            ),
            FEATURE_MORPH_ATTRIBUTE: (
                build.categorical_value(morph_counts),
                sum(morph_counts.values()),
            ),
            FEATURE_MORPH_COVERAGE: (
                build.ratio_value(tokens_with_morph, len(linguistic_tokens))
                if linguistic_tokens
                else None,
                len(linguistic_tokens),
            ),
        }
        assert set(values) == set(feature_ids)
        availability = {
            FEATURE_UPOS: has_pos,
            FEATURE_DEPENDENCY_RELATION: has_dep,
            FEATURE_DEPENDENCY_DISTANCE: has_dep,
            FEATURE_MORPH_ATTRIBUTE: has_morph,
            FEATURE_MORPH_COVERAGE: has_morph,
        }

        observations: list[FeatureObservation] = []
        for feature_id in feature_ids:
            fdef = get_feature(feature_id)
            if not availability[feature_id]:
                observations.append(
                    build.status(
                        fdef,
                        self.analyzer_id,
                        self.implementation_version,
                        FeatureStatus.UNAVAILABLE,
                    )
                )
                continue
            value, support_count = values[feature_id]
            observations.append(
                build.value_observation(
                    fdef, self.analyzer_id, self.implementation_version, value, support_count
                )
            )
        return AnalyzerOutput(observations=tuple(observations))
