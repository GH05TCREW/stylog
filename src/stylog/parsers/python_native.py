"""CPython tokenize/ast parser mechanics (spec 8.1, 8.4, 8.8).

Produces normalized runtime facts only — plain frozen dataclasses, no Pydantic
models, no feature observations. The token stream is atomic (spec 8.4): any
tokenization failure discards all partial tokens and the AST is never
attempted. The AST stage runs only after successful tokenization and inside
the configured resource guards (``max_ast_bytes`` / ``max_ast_nesting``).
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass

from stylog.runtime import RuntimeArtifact

PYTHON_ENCODING_ERROR = "PYTHON_ENCODING_ERROR"
PYTHON_TOKENIZE_ERROR = "PYTHON_TOKENIZE_ERROR"
PYTHON_AST_PARSE_ERROR = "PYTHON_AST_PARSE_ERROR"
PYTHON_AST_RESOURCE_LIMIT = "PYTHON_AST_RESOURCE_LIMIT"
PYTHON_LOCATION_UNAVAILABLE = "PYTHON_LOCATION_UNAVAILABLE"


@dataclass(frozen=True)
class PythonParseFacts:
    """Normalized result of running CPython tokenize/ast over raw source bytes."""

    decoded_text: str | None = None
    encoding: str | None = None
    tokens: list[tokenize.TokenInfo] | None = None
    token_error_code: str | None = None
    tree: ast.Module | None = None
    ast_error_code: str | None = None
    ast_resource_limited: bool = False
    decode_error_code: str | None = None
    max_depth: int | None = None


def _raw_max_depth(tree: ast.AST) -> int:
    """Maximum raw AST edge depth with the Module root at depth 0 (iterative)."""
    max_depth = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return max_depth


def parse_python(artifact: RuntimeArtifact, config: object) -> PythonParseFacts:
    """Run the CPython decode/tokenize/ast pipeline for one Python artifact.

    Failure semantics (spec 8.1, 8.4, 8.8):
    - decode failure: ``decode_error_code`` set, no token/AST work at all;
    - tokenization failure: ``token_error_code`` set, all partial tokens
      discarded, AST never attempted;
    - AST skipped when ``len(raw_bytes) > max_ast_bytes`` or when the parsed
      tree's raw-edge nesting exceeds ``max_ast_nesting``: ``ast_resource_limited``;
    - AST parse failure: ``ast_error_code`` set, tokens stay valid.
    """
    python_config = config.analysis.code.python
    raw = artifact.raw_bytes

    # Spec 8.1: tokenize.detect_encoding exactly (BOM / PEP 263 / conflicts /
    # default UTF-8); generic text encoding never overrides Python source.
    try:
        encoding, _consumed = tokenize.detect_encoding(io.BytesIO(raw).readline)
        decoded = raw.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return PythonParseFacts(decode_error_code=PYTHON_ENCODING_ERROR)

    # Spec 8.4: the token stream is atomic.
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(raw).readline))
    except (tokenize.TokenError, SyntaxError, UnicodeDecodeError):
        return PythonParseFacts(
            decoded_text=decoded,
            encoding=encoding,
            token_error_code=PYTHON_TOKENIZE_ERROR,
        )

    if len(raw) > python_config.max_ast_bytes:
        return PythonParseFacts(
            decoded_text=decoded,
            encoding=encoding,
            tokens=tokens,
            ast_resource_limited=True,
        )

    try:
        tree = ast.parse(decoded, mode="exec", type_comments=False, feature_version=None)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return PythonParseFacts(
            decoded_text=decoded,
            encoding=encoding,
            tokens=tokens,
            ast_error_code=PYTHON_AST_PARSE_ERROR,
        )

    max_depth = _raw_max_depth(tree)
    if max_depth > python_config.max_ast_nesting:
        return PythonParseFacts(
            decoded_text=decoded,
            encoding=encoding,
            tokens=tokens,
            ast_resource_limited=True,
            max_depth=max_depth,
        )

    return PythonParseFacts(
        decoded_text=decoded,
        encoding=encoding,
        tokens=tokens,
        tree=tree,
        max_depth=max_depth,
    )
