"""Artifact descriptors, content identity, and source spans (spec 5.3-5.4, 5.11)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from stylog.domain._base import HexDigest64, PortableModel


class ArtifactKind(StrEnum):
    TEXT = "text"
    CODE = "code"


class ContentIdentitySha256(PortableModel):
    mode: Literal["sha256"] = "sha256"
    sha256: HexDigest64


class ContentIdentitySuppressed(PortableModel):
    mode: Literal["suppressed"] = "suppressed"


ContentIdentity = Annotated[
    ContentIdentitySha256 | ContentIdentitySuppressed,
    Field(discriminator="mode"),
]


class ArtifactDescriptor(PortableModel):
    artifact_id: str
    kind: ArtifactKind
    language: str
    encoding: str
    byte_count: int
    character_count: int
    content_identity: ContentIdentity


class SourcePosition(PortableModel):
    line: int  # 1-based
    column: int  # 0-based Unicode code-point column


class SourceSpan(PortableModel):
    start: SourcePosition
    end: SourcePosition  # exclusive


class EmbeddedArtifactDescriptor(PortableModel):
    artifact: ArtifactDescriptor
    parent_artifact_id: str
    embedded_kind: Literal["comment_block", "inline_comment", "docstring"]
    ordinal: int
    source_span: SourceSpan
    docstring_owner: str | None = None  # omitted unless docstring
