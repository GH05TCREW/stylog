"""Portable JSON/JSONL reading and atomic writing (spec 14.6, 19.11)."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from stylog.exceptions import PortableArtifactError
from stylog.serialization.canonical import canonical_bytes, file_bytes

M = TypeVar("M", bound=BaseModel)


def model_from_bytes(data: bytes, model_type: type[M]) -> M:
    """Parse and validate a portable object, rejecting trailing garbage/nulls."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PortableArtifactError(f"portable JSON is not UTF-8: {exc}") from exc
    stripped = text.rstrip("\n")
    if "\n" in stripped:
        # JSONL-style multi-line input is not a single portable object.
        raise PortableArtifactError("expected a single-line portable JSON object")
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise PortableArtifactError(f"invalid portable JSON: {exc}") from exc
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise PortableArtifactError(f"portable artifact failed validation: {exc}") from exc


def read_json(path: str | Path, model_type: type[M]) -> M:
    data = Path(path).read_bytes()
    return model_from_bytes(data, model_type)


def write_json_atomic(path: str | Path, model: BaseModel, *, force: bool = False) -> None:
    """Atomically write canonical bytes + one LF; refuse overwrite without force."""
    write_bytes_atomic(path, file_bytes(model), force=force)


@contextlib.contextmanager
def atomic_temp_path(target: Path, *, suffix: str = "") -> Iterator[tuple[int, str]]:
    """Yield ``(fd, temp_name)`` for a temp file next to ``target``.

    On clean exit the temp file replaces ``target`` (os.replace); on any
    failure it is removed. Callers own writing the temp file (via ``fd`` or by
    path) and any permission/fsync hardening.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=suffix, dir=str(target.parent)
    )
    try:
        yield fd, temp_name
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_bytes_atomic(path: str | Path, data: bytes, *, force: bool = False) -> None:
    target = Path(path)
    if target.exists() and not force:
        raise PortableArtifactError(f"output exists (use force to overwrite): {target}")
    with atomic_temp_path(target) as (fd, _), os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(target.parent)


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def jsonl_bytes(models: list[BaseModel] | tuple[BaseModel, ...]) -> bytes:
    """One canonical object per line + LF; no blank lines."""
    return b"".join(canonical_bytes(model) + b"\n" for model in models)


def read_jsonl(path: str | Path, model_type: type[M]) -> list[M]:
    out: list[M] = []
    for line in Path(path).read_bytes().split(b"\n"):
        if not line:
            continue
        out.append(model_from_bytes(line, model_type))
    return out
