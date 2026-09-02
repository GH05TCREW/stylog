"""Representation specifications for the 'ml' capability (spec 20.7).

Four representation identities exist in v0.1, all scikit-learn vectorizers
with every constructor parameter stated explicitly (never sklearn defaults):

- ``stylog.representation.char_ngram_count/1`` — char n-grams (3,5), raw counts
- ``stylog.representation.word_ngram_count/1`` — word n-grams (1,3), raw counts
- ``stylog.representation.char_tfidf/1``       — char n-grams (3,5), TF-IDF
- ``stylog.representation.word_tfidf/1``       — word n-grams (1,3), TF-IDF

The ``representation_id`` recorded in portable artifacts is the full id
string *including* the ``/1`` suffix; ``semantic_version`` is ``"1.0.0"``.

Char representations consume the artifact text exactly as decoded: Stylog
applies no normalization, no lowercasing, no accent stripping
(``analyzer="char"``, ``lowercase=False``, ``strip_accents=None``,
``preprocessor=None``). Word representations consume the precomputed Stylog
WORD token sequence, casefolded token by token, through an identity
tokenizer; sklearn's default token regex is never used
(``token_pattern=None``).

The ``params`` tree is JSON-safe so it can be hashed and stored: the string
``"none"`` denotes an explicit Python ``None`` passed to sklearn, and the
string ``"identity"`` denotes the pass-through tokenizer for precomputed
token lists. Every sklearn constructor parameter Stylog passes is recorded
in this tree, so ``fit_config_sha256`` covers the complete vectorizer
configuration.

This module never imports sklearn, numpy, or scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from stylog.exceptions import ConfigurationError

SEMANTIC_VERSION = "1.0.0"
PREPROCESSING_VERSION = "1.0.0"
SCIENTIFIC_COMPATIBILITY_ID = "stylog.sklearn-vectorizers/1"
BACKEND_ID = "scikit-learn"

#: Prefix of the local content-addressed fit-state resource identity. The
#: short form used after the prefix is the representation kind (e.g.
#: ``stylog.representation.state/char_tfidf``).
STATE_RESOURCE_PREFIX = "stylog.representation.state/"

RepresentationKind = Literal["char_ngram_count", "word_ngram_count", "char_tfidf", "word_tfidf"]

REPRESENTATION_KINDS: tuple[RepresentationKind, ...] = (
    "char_ngram_count",
    "word_ngram_count",
    "char_tfidf",
    "word_tfidf",
)

#: Full representation ids (with the "/1" suffix) keyed by kind.
REPRESENTATION_IDS: dict[RepresentationKind, str] = {
    kind: f"stylog.representation.{kind}/1" for kind in REPRESENTATION_KINDS
}

#: CLI --representation tokens keyed by kind (spec 19, `represent` command).
CLI_TOKENS: dict[RepresentationKind, str] = {
    kind: kind.replace("_", "-") for kind in REPRESENTATION_KINDS
}

_VECTORIZER_BASE: dict[str, Any] = {
    "analyzer": "",  # filled per kind below
    "binary": False,
    "decode_error": "strict",
    "dtype": "float64",
    "encoding": "utf-8",
    "input": "content",
    "lowercase": False,
    "max_df": 1.0,
    "max_features": "none",
    "min_df": 1,
    "ngram_range": [0, 0],  # filled per kind below
    "preprocessor": "none",
    "stop_words": "none",
    "strip_accents": "none",
    "token_pattern": "none",  # sklearn's default token regex MUST NOT be used
    "tokenizer": "none",
    "vocabulary": "none",  # fixed vocabulary is injected at transform time
}

_TFIDF_PARAMS: dict[str, Any] = {
    "norm": "l2",
    "smooth_idf": True,
    "sublinear_tf": False,
    "use_idf": True,
}


def _params_for(kind: RepresentationKind) -> dict[str, Any]:
    params = dict(_VECTORIZER_BASE)
    if kind.startswith("char_"):
        params["analyzer"] = "char"
        params["ngram_range"] = [3, 5]
    else:
        params["analyzer"] = "word"
        params["ngram_range"] = [1, 3]
        params["tokenizer"] = "identity"  # precomputed Stylog WORD token lists
    if kind.endswith("_tfidf"):
        params.update(_TFIDF_PARAMS)
    return params


@dataclass(frozen=True)
class RepresentationSpec:
    """Frozen specification of one Stylog representation (spec 20.7).

    ``representation_id`` is the full id including the ``/1`` suffix.
    ``params`` is the explicit, JSON-safe sklearn vectorizer parameter tree
    (see module docstring for the ``"none"``/``"identity"`` sentinels); it is
    treated as immutable.
    """

    representation_id: str
    kind: RepresentationKind
    params: dict[str, Any] = field(compare=True)

    @property
    def semantic_version(self) -> str:
        return SEMANTIC_VERSION

    @property
    def short_id(self) -> str:
        """The representation id without the ``/1`` version suffix."""
        return self.representation_id.removesuffix("/1")

    @property
    def cli_token(self) -> str:
        return CLI_TOKENS[self.kind]

    @property
    def is_word(self) -> bool:
        return self.kind.startswith("word_")

    @property
    def is_tfidf(self) -> bool:
        return self.kind.endswith("_tfidf")

    @property
    def state_resource_id(self) -> str:
        """Resource id of the local content-addressed fitted state."""
        return STATE_RESOURCE_PREFIX + self.kind

    def fit_config_tree(self) -> dict[str, Any]:
        """The exact tree hashed into ``RepresentationFit.fit_config_sha256``."""
        return {
            "params": self.params,
            "representation_id": self.representation_id,
            "semantic_version": self.semantic_version,
        }


SPECS: dict[RepresentationKind, RepresentationSpec] = {
    kind: RepresentationSpec(
        representation_id=REPRESENTATION_IDS[kind],
        kind=kind,
        params=_params_for(kind),
    )
    for kind in REPRESENTATION_KINDS
}


def representation_spec(ref: str | RepresentationSpec) -> RepresentationSpec:
    """Resolve a representation reference to its frozen spec.

    Accepts a RepresentationSpec (returned unchanged), the full id
    (``"stylog.representation.char_tfidf/1"``), the id without the version
    suffix (``"stylog.representation.char_tfidf"``), the bare kind
    (``"char_tfidf"``), or the CLI token (``"char-tfidf"``).
    """
    if isinstance(ref, RepresentationSpec):
        return ref
    key = ref.removesuffix("/1")
    key = key.removeprefix("stylog.representation.")
    key = key.replace("-", "_")
    spec = SPECS.get(key)  # type: ignore[arg-type]
    if spec is None:
        raise ConfigurationError(f"unknown representation id: {ref!r}")
    return spec
