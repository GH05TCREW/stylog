"""Input adapters: bytes/text/file/stdin -> RuntimeArtifact (spec 18.1-18.8).

These adapters convert inputs to runtime artifacts and cease being
architecturally interesting after ingestion. Decoding is always strict; raw
byte identity always covers the exact original bytes.
"""

from __future__ import annotations

import codecs
import io
import tokenize
from pathlib import Path

from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.exceptions import DecodeError, InputError, ResourceLimitError, UnsupportedInputError
from stylog.runtime import RuntimeArtifact
from stylog.serialization.canonical import sha256_hex

EXTENSION_KIND_LANGUAGE: dict[str, tuple[str, str]] = {
    ".py": ("code", "python"),
    ".js": ("code", "javascript"),
    ".mjs": ("code", "javascript"),
    ".cjs": ("code", "javascript"),
    ".jsx": ("code", "javascript"),
    ".ts": ("code", "typescript"),
    ".tsx": ("code", "typescript"),
    ".c": ("code", "c"),
    ".rs": ("code", "rust"),
    ".txt": ("text", "und"),
    ".md": ("text", "und"),
    ".rst": ("text", "und"),
}


def _check_decoded(text: str, *, source: str) -> str:
    if "\x00" in text:
        raise InputError(f"decoded input contains U+0000 (INPUT_NUL): {source}")
    return text


def decode_python_bytes(raw: bytes, *, source: str = "<bytes>") -> tuple[str, str]:
    """Exact CPython encoding detection contract (spec 8.1)."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        raise DecodeError(f"Python encoding detection failed (PYTHON_ENCODING_ERROR): {exc}") from exc
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise DecodeError(f"Python source decode failed (PYTHON_ENCODING_ERROR): {exc}") from exc
    return _check_decoded(text, source=source), encoding


def decode_text_bytes(raw: bytes, encoding: str, *, source: str = "<bytes>") -> tuple[str, str]:
    """Strict text decoding; UTF-8 BOM uses utf-8-sig while identity keeps the BOM."""
    try:
        codec = codecs.lookup(encoding)
    except LookupError as exc:
        raise UnsupportedInputError(f"unknown text encoding: {encoding!r}") from exc
    effective = codec.name
    if effective == "utf-8" and raw.startswith(codecs.BOM_UTF8):
        effective = "utf-8-sig"
    try:
        text = raw.decode(effective, errors="strict")
    except UnicodeDecodeError as exc:
        raise DecodeError(f"text decode failed (INPUT_DECODE_ERROR): {exc}") from exc
    return _check_decoded(text, source=source), effective


def artifact_from_bytes(
    data: bytes,
    *,
    artifact_id: str,
    kind: str,
    language: str,
    encoding: str = "utf-8",
    config: StylogConfig | None = None,
) -> RuntimeArtifact:
    if len(data) > (config.input.max_file_bytes if config else 8_388_608):
        raise ResourceLimitError(
            f"input exceeds max_file_bytes ({len(data)} bytes): {artifact_id} (INPUT_TOO_LARGE)"
        )
    if kind not in ("text", "code"):
        raise UnsupportedInputError(f"unsupported artifact kind: {kind!r}")
    if kind == "code" and language == "python":
        text, used_encoding = decode_python_bytes(data, source=artifact_id)
    else:
        text, used_encoding = decode_text_bytes(data, encoding, source=artifact_id)
    return RuntimeArtifact(
        artifact_id=artifact_id,
        kind=ArtifactKind(kind),
        language=language,
        encoding=used_encoding,
        raw_bytes=data,
        text=text,
        content_sha256=sha256_hex(data),
    )


def artifact_from_text(
    text: str,
    *,
    artifact_id: str,
    language: str = "und",
) -> RuntimeArtifact:
    """In-memory string input: reject lone surrogates; identity is strict UTF-8."""
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        raise InputError(f"in-memory text contains lone surrogates (INPUT_SURROGATE): {artifact_id}")
    _check_decoded(text, source=artifact_id)
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id=artifact_id,
        kind=ArtifactKind.TEXT,
        language=language,
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=sha256_hex(raw),
    )


def infer_kind_language(
    path: Path,
    *,
    kind: str = "auto",
    language: str = "auto",
    default_language: str = "und",
) -> tuple[str, str]:
    """Extension mapping per spec 18.8; explicit overrides win; nothing is guessed."""
    suffix = path.suffix.lower()
    mapped = EXTENSION_KIND_LANGUAGE.get(suffix)
    if kind == "auto":
        if mapped is None:
            raise UnsupportedInputError(
                f"cannot infer artifact kind/language from extension {suffix!r} (INPUT_UNSUPPORTED)"
            )
        kind = mapped[0]
    if language == "auto":
        if mapped is not None:
            language = mapped[1]
        elif kind == "text":
            language = default_language
        else:
            raise UnsupportedInputError(
                f"explicit --language is required for {suffix!r} (INPUT_UNSUPPORTED)"
            )
    return kind, language


def artifact_from_file(
    path: Path,
    *,
    artifact_id: str,
    kind: str = "auto",
    language: str = "auto",
    config: StylogConfig,
) -> RuntimeArtifact:
    if path.is_symlink():
        raise UnsupportedInputError(f"symlink input rejected (SYMLINK_REJECTED): {path.name}")
    if not path.is_file():
        raise InputError(f"input not found (INPUT_NOT_FOUND): {path.name}")
    resolved_kind, resolved_language = infer_kind_language(
        path, kind=kind, language=language, default_language=config.analysis.language
    )
    raw = path.read_bytes()
    encoding = config.input.text_encoding if resolved_kind == "text" else "utf-8"
    return artifact_from_bytes(
        raw,
        artifact_id=artifact_id,
        kind=resolved_kind,
        language=resolved_language,
        encoding=encoding,
        config=config,
    )


def artifact_from_stdin(
    data: bytes,
    *,
    artifact_id: str = "stdin",
    kind: str = "text",
    language: str = "und",
    encoding: str = "utf-8",
    config: StylogConfig,
) -> RuntimeArtifact:
    return artifact_from_bytes(
        data,
        artifact_id=artifact_id,
        kind=kind,
        language=language,
        encoding=encoding,
        config=config,
    )
