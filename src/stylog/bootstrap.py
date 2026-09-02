"""Default local service construction (spec 20.4). No global singleton."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_path

from stylog.analysis.registry import (
    FUNCTION_WORDS_EN_RESOURCE_ID,
    FUNCTION_WORDS_EN_SHA256,
)
from stylog.capability import require_capability
from stylog.config import StylogConfig, parse_no_cache_env
from stylog.domain.provenance import ResourceSignature, current_runtime_signature
from stylog.exceptions import ResourceError
from stylog.infrastructure.baselines import FilesystemBaselineResolver
from stylog.infrastructure.cache import FilesystemCacheStore, NullCacheStore
from stylog.infrastructure.paths import store_root
from stylog.infrastructure.resources import PackageResourceResolver
from stylog.ports import ResourceRequest, RuntimeServices
from stylog.runtime import AnalysisContext, ResourceHandles


def default_cache_root() -> Path:
    return store_root("STYLOG_CACHE_DIR", user_cache_path("stylog") / "v1")


def build_default_services(
    config: StylogConfig,
    *,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    env: dict[str, str] | None = None,
) -> RuntimeServices:
    env = os.environ if env is None else env
    cache_enabled = config.cache.enabled and not no_cache
    if "STYLOG_NO_CACHE" in env and parse_no_cache_env(env["STYLOG_NO_CACHE"]):
        cache_enabled = False
    if cache_enabled:
        cache = FilesystemCacheStore(cache_dir or default_cache_root())
    else:
        cache = NullCacheStore()
    return RuntimeServices(
        cache=cache,
        baselines=FilesystemBaselineResolver(config),
        resources=PackageResourceResolver(),
    )


def build_resource_handles(
    config: StylogConfig,
    services: RuntimeServices,
    *,
    nlp_model_name: str | None = None,
) -> ResourceHandles:
    """Resolve package resources (hash-verified) and the explicit NLP model."""
    function_words: frozenset[str] | None = None
    function_words_sig: ResourceSignature | None = None
    if config.analysis.text.function_words_en:
        try:
            resolved = services.resources.resolve(ResourceRequest(FUNCTION_WORDS_EN_RESOURCE_ID))
        except ResourceError:
            resolved = None  # resource absent: features degrade to typed unavailability
        if resolved is not None and resolved.signature.sha256 != FUNCTION_WORDS_EN_SHA256:
            raise ResourceError(
                "RESOURCE_MISMATCH: function-words resource hash differs from the pinned "
                "stylog.function_words.en/1.0.0 identity"
            )
        if resolved is not None:
            assert resolved.data is not None
            function_words = frozenset(resolved.data.decode("utf-8").splitlines())
            function_words_sig = resolved.signature

    mappings = {}
    manifest = {}
    manifest_sha256 = None
    if config.analysis.code.tree_sitter.enabled:
        from stylog.parsers.tree_sitter import (
            load_manifest,
            load_manifest_sha256,
            load_mappings,
        )

        mappings = load_mappings(services.resources)
        manifest = load_manifest(services.resources)
        manifest_sha256 = load_manifest_sha256(services.resources)

    nlp_model = None
    nlp_backend = None
    if nlp_model_name:
        linguistic = require_capability("stylog.analysis.linguistic", "nlp")
        nlp_model, nlp_backend = linguistic.load_spacy_model(nlp_model_name, services)

    return ResourceHandles(
        function_words_en=function_words,
        function_words_en_signature=function_words_sig,
        tree_sitter_mappings=mappings,
        grammar_manifest=manifest,
        grammar_manifest_sha256=manifest_sha256,
        nlp_model=nlp_model,
        nlp_model_backend=nlp_backend,
    )


def build_context(
    config: StylogConfig,
    services: RuntimeServices | None = None,
    *,
    nlp_model_name: str | None = None,
) -> AnalysisContext:
    if services is None:
        services = build_default_services(config)
    return AnalysisContext(
        config=config,
        runtime=current_runtime_signature(),
        resources=build_resource_handles(config, services, nlp_model_name=nlp_model_name),
    )
