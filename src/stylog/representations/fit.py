"""Fit and transform for scikit-learn Representations (spec 20.7).

The 'ml' capability: sklearn, numpy, and scipy are imported lazily inside
functions only — importing this module never imports them (release gate).

Fitted state is a LOCAL content-addressed resource, never a network one
(spec 22). State files live under
``platformdirs.user_data_path("stylog")/"fits"/<state_sha256>.json`` and hold
the canonical JCS bytes plus exactly one LF of a JSON object::

    {"representation_id": ..., "semantic_version": ..., "params": {...},
     "vocabulary": [terms in Unicode scalar lexical order],
     "idf": [floats in the same order]}   # "idf" absent for count kinds

The state sha256 is the SHA-256 of those canonical bytes (trailing LF
excluded), identical to the state's content-addressed file name.
``STYLOG_FITS_DIR`` overrides the store root (local test/ops override,
mirroring the ``STYLOG_CACHE_DIR`` precedent).

Fit corpus manifest hash: ``sha256_of_tree({"subjects": [...]})`` where the
list holds each corpus artifact's ``content_sha256`` in deterministic
subject-ID order (artifacts sorted by ``artifact_id``, Unicode scalar
lexical order). Fit config hash: ``sha256_of_tree(spec.fit_config_tree())``.
"""

from __future__ import annotations

import json
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_data_path

from stylog.analysis.text import tokenize_text
from stylog.capability import require_capability
from stylog.domain._base import is_sorted_unique
from stylog.domain.provenance import BackendSignature, PackageProvenance
from stylog.domain.representation import (
    Representation,
    RepresentationFit,
    RepresentationResourceSignature,
    SparseCoordinate,
    SparseVectorValue,
)
from stylog.exceptions import PortableArtifactError, ResourceError
from stylog.infrastructure.paths import store_root
from stylog.representations.spec import (
    BACKEND_ID,
    PREPROCESSING_VERSION,
    SCIENTIFIC_COMPATIBILITY_ID,
    RepresentationSpec,
    representation_spec,
)
from stylog.serialization.canonical import canonical_bytes_of_tree, sha256_hex, sha256_of_tree
from stylog.serialization.jsonio import read_json, write_bytes_atomic, write_json_atomic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stylog.config import StylogConfig
    from stylog.ports import RuntimeServices
    from stylog.runtime import RuntimeArtifact


def _ml_imports() -> tuple[Any, Any, Any, Any]:
    """Import the optional 'ml' stack lazily; translate absence to a typed error."""
    np = require_capability("numpy", "ml")
    sklearn_text = require_capability("sklearn.feature_extraction.text", "ml")
    return (
        np,
        sklearn_text.CountVectorizer,
        sklearn_text.TfidfTransformer,
        sklearn_text.TfidfVectorizer,
    )


def _identity_tokens(doc: Any) -> Any:
    """Pass-through tokenizer for precomputed Stylog WORD token lists."""
    return doc


def _fits_root() -> Path:
    return store_root("STYLOG_FITS_DIR", user_data_path("stylog") / "fits")


def _from_tree(value: Any) -> Any:
    """Map the JSON-safe params-tree sentinel "none" back to explicit None."""
    return None if value == "none" else value


def _vectorizer_kwargs(spec: RepresentationSpec, np: Any) -> dict[str, Any]:
    """Translate a spec's params tree into explicit sklearn constructor kwargs.

    Every parameter is stated explicitly (never sklearn defaults); the params
    tree is the single source of truth, so ``fit_config_sha256`` covers the
    complete vectorizer configuration.
    """
    params = spec.params
    if params["dtype"] != "float64":
        raise ResourceError(f"unsupported representation dtype: {params['dtype']!r}")
    tokenizer = _identity_tokens if params["tokenizer"] == "identity" else None
    kwargs: dict[str, Any] = {
        "input": params["input"],
        "encoding": params["encoding"],
        "decode_error": params["decode_error"],
        "strip_accents": _from_tree(params["strip_accents"]),
        "lowercase": params["lowercase"],
        "preprocessor": _from_tree(params["preprocessor"]),
        "tokenizer": tokenizer,
        "stop_words": _from_tree(params["stop_words"]),
        "token_pattern": _from_tree(params["token_pattern"]),
        "ngram_range": (params["ngram_range"][0], params["ngram_range"][1]),
        "analyzer": params["analyzer"],
        "max_df": params["max_df"],
        "min_df": params["min_df"],
        "max_features": _from_tree(params["max_features"]),
        "vocabulary": _from_tree(params["vocabulary"]),
        "binary": params["binary"],
        "dtype": np.float64,
    }
    if spec.is_tfidf:
        kwargs["norm"] = params["norm"]
        kwargs["use_idf"] = params["use_idf"]
        kwargs["smooth_idf"] = params["smooth_idf"]
        kwargs["sublinear_tf"] = params["sublinear_tf"]
    return kwargs


def _document(text: str, spec: RepresentationSpec) -> str | list[str]:
    """Build the sklearn input document for one subject (spec 20.7).

    Char kinds: the text exactly as decoded. Word kinds: the Stylog WORD
    token sequence, casefolded token by token (numbers are not words).
    """
    if spec.is_word:
        return [t.text.casefold() for t in tokenize_text(text) if t.kind == "word"]
    return text


def _backend_signature() -> BackendSignature:
    """Dependency-aware backend provenance for the sklearn vectorizers."""
    sklearn = require_capability("sklearn", "ml")
    packages = tuple(
        sorted(
            (
                PackageProvenance(package="numpy", version=package_version("numpy")),
                PackageProvenance(package="scikit-learn", version=package_version("scikit-learn")),
                PackageProvenance(package="scipy", version=package_version("scipy")),
            ),
            key=lambda package: package.package,
        )
    )
    return BackendSignature(
        backend_id=BACKEND_ID,
        implementation_version=sklearn.__version__,
        scientific_compatibility_id=SCIENTIFIC_COMPATIBILITY_ID,
        packages=packages,
    )


def _store_state(state_tree: dict[str, Any]) -> tuple[str, Path]:
    """Write the canonical state resource; return (state_sha256, path)."""
    state_bytes = canonical_bytes_of_tree(state_tree)
    state_sha256 = sha256_hex(state_bytes)
    state_path = _fits_root() / f"{state_sha256}.json"
    if not state_path.exists():
        try:
            write_bytes_atomic(state_path, state_bytes + b"\n", force=False)
        except PortableArtifactError:
            # Concurrent writer stored byte-identical content first; the
            # content address makes this benign.
            pass
    return state_sha256, state_path


def fit_representation(
    spec: RepresentationSpec | str,
    corpus: Sequence[RuntimeArtifact],
    *,
    config: StylogConfig | None = None,
    services: RuntimeServices | None = None,
) -> RepresentationFit:
    """Fit a representation on a corpus and store its state locally.

    The corpus is processed in deterministic subject-ID order (sorted by
    ``artifact_id``). ``config``/``services`` are accepted for application
    symmetry; the 'ml' capability has no config-affected semantics and the
    fit store is a plain local directory, so neither is consulted.
    """
    np, CountVectorizer, _, TfidfVectorizer = _ml_imports()
    spec = representation_spec(spec)
    subjects = sorted(corpus, key=lambda artifact: artifact.artifact_id)
    documents = [_document(artifact.text, spec) for artifact in subjects]

    kwargs = _vectorizer_kwargs(spec, np)
    vectorizer = TfidfVectorizer(**kwargs) if spec.is_tfidf else CountVectorizer(**kwargs)
    vectorizer.fit(documents)

    # Canonicalize the vocabulary into Unicode scalar lexical term order
    # (Python str comparison) and reorder the IDF state to match.
    sklearn_vocabulary = vectorizer.vocabulary_
    terms = sorted(sklearn_vocabulary)
    idf: list[float] | None = None
    if spec.is_tfidf:
        sklearn_idf = vectorizer.idf_
        idf = [float(sklearn_idf[sklearn_vocabulary[term]]) for term in terms]

    state_tree: dict[str, Any] = {
        "representation_id": spec.representation_id,
        "semantic_version": spec.semantic_version,
        "params": spec.params,
        "vocabulary": terms,
    }
    if idf is not None:
        state_tree["idf"] = idf
    state_sha256, _ = _store_state(state_tree)

    manifest_sha256 = sha256_of_tree(
        {"subjects": [artifact.content_sha256 for artifact in subjects]}
    )
    return RepresentationFit(
        fit_id=state_sha256,
        representation_id=spec.representation_id,
        representation_semantic_version=spec.semantic_version,
        source_manifest_sha256=manifest_sha256,
        fit_config_sha256=sha256_of_tree(spec.fit_config_tree()),
        state_resource=RepresentationResourceSignature(
            resource_id=spec.state_resource_id,
            resource_version="1.0.0",
            sha256=state_sha256,
        ),
        backend=_backend_signature(),
    )


def _load_state(fit: RepresentationFit) -> tuple[list[str], list[float] | None]:
    """Load and verify the local fit state; no network resolution, ever."""
    spec = representation_spec(fit.representation_id)
    sha256 = fit.state_resource.sha256
    state_path = _fits_root() / f"{sha256}.json"
    if not state_path.is_file():
        raise ResourceError(
            f"representation fit state not provisioned locally: {state_path.name} "
            f"for {fit.representation_id!r} (fit states are local content-addressed "
            "resources; re-run the fit on this machine)"
        )
    raw = state_path.read_bytes()
    content = raw[:-1] if raw.endswith(b"\n") else raw
    if sha256_hex(content) != sha256:
        raise ResourceError(
            f"RESOURCE_MISMATCH: representation state {state_path.name} fails its "
            "content-addressed hash check"
        )
    try:
        state = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceError(f"invalid representation state {state_path.name}: {exc}") from exc
    if not isinstance(state, dict):
        raise ResourceError(f"representation state {state_path.name} is not a JSON object")
    if state.get("representation_id") != fit.representation_id:
        raise ResourceError(
            f"RESOURCE_MISMATCH: representation state is for "
            f"{state.get('representation_id')!r}, not {fit.representation_id!r}"
        )
    if state.get("semantic_version") != fit.representation_semantic_version:
        raise ResourceError(
            "RESOURCE_MISMATCH: representation state semantic version mismatch for "
            f"{fit.representation_id!r}"
        )
    vocabulary = state.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or not all(isinstance(term, str) for term in vocabulary)
        or not is_sorted_unique(vocabulary)
    ):
        raise ResourceError(
            f"representation state {state_path.name} has a non-canonical vocabulary"
        )
    idf = state.get("idf")
    if spec.is_tfidf:
        if (
            not isinstance(idf, list)
            or len(idf) != len(vocabulary)
            or not all(isinstance(value, (int, float)) for value in idf)
        ):
            raise ResourceError(f"representation state {state_path.name} has an invalid idf vector")
        return vocabulary, [float(value) for value in idf]
    if idf is not None:
        raise ResourceError(
            f"representation state {state_path.name} carries idf for a count representation"
        )
    return vocabulary, None


def _prepare_transform(
    fit_or_spec: RepresentationFit | RepresentationSpec | str,
) -> tuple[RepresentationFit, RepresentationSpec, list[str], Any]:
    """Validate the fit and rebuild its fitted vectorizer (once per call)."""
    np, CountVectorizer, _, TfidfVectorizer = _ml_imports()
    if isinstance(fit_or_spec, (RepresentationSpec, str)):
        spec = representation_spec(fit_or_spec)
        raise ResourceError(
            f"representation {spec.representation_id!r} requires a fitted "
            "state: pass a RepresentationFit (v0.1 defines no fit-free "
            "representations)"
        )
    spec = representation_spec(fit_or_spec.representation_id)
    terms, idf = _load_state(fit_or_spec)
    vectorizer = _build_vectorizer(spec, terms, idf, CountVectorizer, TfidfVectorizer, np)
    return fit_or_spec, spec, terms, vectorizer


def _transform_one(
    vectorizer: Any,
    spec: RepresentationSpec,
    text: str,
    ref: str,
    fit: RepresentationFit,
    terms: list[str],
    backend: BackendSignature,
) -> Representation:
    matrix = vectorizer.transform([_document(text, spec)]).tocoo()
    entries = tuple(
        SparseCoordinate(index=int(column), value=float(value))
        for column, value in sorted(zip(matrix.col, matrix.data, strict=True))
        if value != 0.0
    )
    return Representation(
        subject_ref=ref,
        representation_id=spec.representation_id,
        semantic_version=spec.semantic_version,
        preprocessing_version=PREPROCESSING_VERSION,
        fit_id=fit.fit_id,
        backend=backend,
        resources=(fit.state_resource,),
        value=SparseVectorValue(dimension=len(terms), entries=entries),
    )


def transform_representation(
    fit_or_spec: RepresentationFit | RepresentationSpec | str,
    subject: RuntimeArtifact | str,
    *,
    config: StylogConfig | None = None,
    services: RuntimeServices | None = None,
    subject_ref: str | None = None,
) -> Representation:
    """Transform one subject into a sparse portable Representation.

    ``fit_or_spec`` must be a RepresentationFit whose state is provisioned in
    the local fit store; a bare RepresentationSpec or representation id string
    is fit-free usage, which v0.1 does not define, so it raises ResourceError
    (an unknown id raises ConfigurationError first). The portable result
    carries only coordinates/values plus signatures — never vocabulary terms
    or source text.
    """
    fit, spec, terms, vectorizer = _prepare_transform(fit_or_spec)
    if isinstance(subject, str):
        text = subject
        ref = subject_ref if subject_ref is not None else "text"
    else:
        text = subject.text
        ref = subject_ref if subject_ref is not None else subject.artifact_id
    return _transform_one(vectorizer, spec, text, ref, fit, terms, _backend_signature())


def _build_vectorizer(spec, terms, idf, CountVectorizer, TfidfVectorizer, np):
    """Rebuild the fitted vectorizer from state (once per fit, not per call)."""
    kwargs = _vectorizer_kwargs(spec, np)
    kwargs["vocabulary"] = {term: index for index, term in enumerate(terms)}
    if spec.is_tfidf:
        vectorizer = TfidfVectorizer(**kwargs)
        assert idf is not None
        vectorizer.idf_ = np.asarray(idf, dtype=np.float64)
    else:
        vectorizer = CountVectorizer(**kwargs)
    return vectorizer


def transform_many(
    fit_or_spec: RepresentationFit | RepresentationSpec | str,
    subjects: Sequence[RuntimeArtifact | str],
    *,
    config: StylogConfig | None = None,
    services: RuntimeServices | None = None,
) -> list[Representation]:
    """Transform many subjects with one loaded fit state (bulk path).

    Semantically identical to calling :func:`transform_representation` per
    subject; the fitted state and vectorizer are loaded/built exactly once.
    """
    fit, spec, terms, vectorizer = _prepare_transform(fit_or_spec)
    backend = _backend_signature()

    out: list[Representation] = []
    for subject in subjects:
        if isinstance(subject, str):
            text = subject
            ref = "text"
        else:
            text = subject.text
            ref = subject.artifact_id
        out.append(_transform_one(vectorizer, spec, text, ref, fit, terms, backend))
    return out


def fit_representation_cli(
    representation_id: str,
    artifacts: list[RuntimeArtifact],
    fit_output_path: str | Path,
    *,
    force: bool = False,
) -> RepresentationFit:
    """CLI helper: fit and atomically write the RepresentationFit."""
    fit = fit_representation(representation_id, artifacts)
    write_json_atomic(fit_output_path, fit, force=force)
    return fit


def transform_representation_cli(
    fit_path: str | Path,
    artifacts: list[RuntimeArtifact],
) -> list[Representation]:
    """CLI helper: read a RepresentationFit and transform each artifact."""
    fit = read_json(fit_path, RepresentationFit)
    return transform_many(fit, artifacts)
