"""Provenance and compatibility signatures (spec sections 5.5 and 5.17)."""

from __future__ import annotations

import platform
import sys
import unicodedata

from pydantic import model_validator

from stylog.domain._base import HexDigest64, PortableModel, is_sorted_unique, tuple_of


class ResourceSignature(PortableModel):
    id: str
    version: str
    sha256: HexDigest64


class RuntimeSignature(PortableModel):
    python_implementation: str
    python_version: str
    python_cache_tag: str
    unicode_database_version: str


def current_runtime_signature() -> RuntimeSignature:
    return RuntimeSignature(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_cache_tag=sys.implementation.cache_tag or "",
        unicode_database_version=unicodedata.unidata_version,
    )


class PackageProvenance(PortableModel):
    package: str
    version: str


class ParserGrammarSignature(PortableModel):
    language: str
    grammar_id: str
    grammar_version: str
    grammar_revision: str
    node_types_sha256: HexDigest64
    grammar_manifest_sha256: HexDigest64
    language_abi_version: int


class ModelSignature(PortableModel):
    model_id: str
    model_revision: str
    model_tree_sha256: HexDigest64
    tokenizer_id: str
    tokenizer_version: str
    tokenizer_tree_sha256: HexDigest64
    preprocessing_id: str
    preprocessing_version: str


class BackendSignature(PortableModel):
    backend_id: str
    implementation_version: str
    scientific_compatibility_id: str
    packages: tuple_of(PackageProvenance) = ()
    resources: tuple_of(ResourceSignature) = ()
    parser_grammar: ParserGrammarSignature | None = None  # omitted unless parser-backed
    model: ModelSignature | None = None  # omitted unless model-backed

    @model_validator(mode="after")
    def _sorted(self) -> BackendSignature:
        package_ids = [package.package for package in self.packages]
        if not is_sorted_unique(package_ids):
            raise ValueError("backend packages must be sorted by unique package name")
        resource_ids = [resource.id for resource in self.resources]
        if not is_sorted_unique(resource_ids):
            raise ValueError("backend resources must be sorted by unique resource id")
        return self


class AnalyzerSignature(PortableModel):
    analyzer_id: str
    implementation_version: str
    feature_registry_version: str
    backend: BackendSignature
    resources: tuple_of(ResourceSignature) = ()

    @model_validator(mode="after")
    def _sorted(self) -> AnalyzerSignature:
        resource_ids = [resource.id for resource in self.resources]
        if not is_sorted_unique(resource_ids):
            raise ValueError("analyzer resources must be sorted by unique resource id")
        return self
