"""The three environmental ports (spec 4.6). Nothing else is injected.

Infrastructure implements these protocols; the scientific core never imports
concrete infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stylog.domain.baseline import Baseline
from stylog.domain.provenance import ResourceSignature


class CacheStore(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, canonical_bytes: bytes) -> None: ...


class BaselineResolver(Protocol):
    def resolve(self, baseline_ref: str) -> Baseline: ...


@dataclass(frozen=True)
class ResourceRequest:
    """A request for a local non-code resource (model, mapping, ...)."""

    resource_id: str
    version: str | None = None
    purpose: str = ""


@dataclass(frozen=True)
class ResolvedResource:
    """Runtime handle for a resolved resource. ``local_path`` is never portable."""

    signature: ResourceSignature
    data: bytes | None = None
    local_path: str | None = None


class ResourceResolver(Protocol):
    def resolve(self, request: ResourceRequest) -> ResolvedResource: ...


@dataclass(frozen=True)
class RuntimeServices:
    """Small internal grouping of the three ports (not a DI framework)."""

    cache: CacheStore
    baselines: BaselineResolver
    resources: ResourceResolver
