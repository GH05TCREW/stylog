"""Scientific cache conformance (spec 17, 25.21)."""

from __future__ import annotations

from stylog.application.fingerprint import fingerprint_artifact
from stylog.bootstrap import build_context
from stylog.config import StylogConfig
from stylog.domain import Fingerprint
from stylog.infrastructure.baselines import StaticBaselineResolver
from stylog.infrastructure.cache import (
    FilesystemCacheStore,
    MemoryCacheStore,
    fingerprint_cache_key,
)
from stylog.infrastructure.ingest import artifact_from_text
from stylog.infrastructure.resources import PackageResourceResolver
from stylog.ports import RuntimeServices
from stylog.serialization.canonical import scientific_sha256
from stylog.serialization.jsonio import model_from_bytes


def _services(cache) -> RuntimeServices:
    return RuntimeServices(
        cache=cache, baselines=StaticBaselineResolver({}), resources=PackageResourceResolver()
    )


def _text_artifact(text: str, artifact_id: str = "t"):
    return artifact_from_text(text, artifact_id=artifact_id, language="und")


def test_same_content_different_id_same_key() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config = StylogConfig()
    ctx = build_context(config, services)
    a = _text_artifact("hello world", "a")
    b = _text_artifact("hello world", "b")
    r1 = fingerprint_artifact(a, config=config, services=services, ctx=ctx)
    r2 = fingerprint_artifact(b, config=config, services=services, ctx=ctx)
    assert not r1.cache_hit
    assert r2.cache_hit  # same content -> same key
    assert len(cache.objects) == 1


def test_scientific_config_change_misses() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config1 = StylogConfig()
    ctx1 = build_context(config1, services)
    artifact = _text_artifact("hello world")
    r1 = fingerprint_artifact(artifact, config=config1, services=services, ctx=ctx1)
    assert not r1.cache_hit
    # disable the window feature -> scientific config changes -> miss
    config2 = StylogConfig.model_validate({"analysis": {"text": {"window_ttr_100": False}}})
    ctx2 = build_context(config2, services)
    r2 = fingerprint_artifact(artifact, config=config2, services=services, ctx=ctx2)
    assert not r2.cache_hit
    assert len(cache.objects) == 2


def test_content_hash_suppression_same_internal_key() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config1 = StylogConfig()
    ctx1 = build_context(config1, services)
    artifact = _text_artifact("hello world")
    r1 = fingerprint_artifact(artifact, config=config1, services=services, ctx=ctx1)
    assert r1.fingerprint.artifact.content_identity.mode == "sha256"
    config2 = StylogConfig.model_validate({"analysis": {"export_content_hashes": False}})
    ctx2 = build_context(config2, services)
    r2 = fingerprint_artifact(artifact, config=config2, services=services, ctx=ctx2)
    assert r2.cache_hit  # suppression is export-only; internal key unchanged
    assert r2.fingerprint.artifact.content_identity.mode == "suppressed"
    assert len(cache.objects) == 1


def test_analyzer_version_change_misses() -> None:
    base = {
        "content_sha256": "0" * 64,
        "kind": "text",
        "language": "und",
        "analysis_config_sha256": "1" * 64,
        "schema_version": "0.1.0",
        "analyzer_ids_versions": (("a", "1.0.0"),),
        "resource_signatures": (),
        "runtime_fields": (),
    }
    key1 = fingerprint_cache_key(**base)
    key2 = fingerprint_cache_key(**{**base, "analyzer_ids_versions": (("a", "1.0.1"),)})
    assert key1 != key2


def test_language_and_kind_change_miss() -> None:
    # kind/language are scientific identity: language-gated analyzers produce
    # different observations per language, so entries must never be shared.
    base = {
        "content_sha256": "0" * 64,
        "kind": "text",
        "language": "en",
        "analysis_config_sha256": "1" * 64,
        "schema_version": "0.1.0",
        "analyzer_ids_versions": (),
        "resource_signatures": (),
        "runtime_fields": (),
    }
    key1 = fingerprint_cache_key(**base)
    assert key1 != fingerprint_cache_key(**{**base, "language": "und"})
    assert key1 != fingerprint_cache_key(**{**base, "language": "fr"})
    assert key1 != fingerprint_cache_key(**{**base, "kind": "code"})


def test_same_content_different_language_no_wrong_hit() -> None:
    # regression: identical bytes analyzed as en vs und must not collide;
    # function-word observations differ by language.
    cache = MemoryCacheStore()
    services = _services(cache)
    config = StylogConfig()
    ctx = build_context(config, services)
    english = artifact_from_text("I would go to the house, but she would not go.", artifact_id="t", language="en")
    und = artifact_from_text("I would go to the house, but she would not go.", artifact_id="t", language="und")
    r_en = fingerprint_artifact(english, config=config, services=services, ctx=ctx)
    r_und = fingerprint_artifact(und, config=config, services=services, ctx=ctx)
    assert not r_en.cache_hit
    assert not r_und.cache_hit
    assert r_en.fingerprint.artifact.language == "en"
    assert r_und.fingerprint.artifact.language == "und"
    assert len(cache.objects) == 2


def test_runtime_change_misses() -> None:
    base = {
        "content_sha256": "0" * 64,
        "kind": "text",
        "language": "und",
        "analysis_config_sha256": "1" * 64,
        "schema_version": "0.1.0",
        "analyzer_ids_versions": (),
        "resource_signatures": (),
        "runtime_fields": (("python_version", "3.14.3"),),
    }
    key1 = fingerprint_cache_key(**base)
    key2 = fingerprint_cache_key(
        **{**base, "runtime_fields": (("python_version", "3.13.0"),)}
    )
    assert key1 != key2


def test_corrupt_cache_entry_recomputed() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config = StylogConfig()
    ctx = build_context(config, services)
    artifact = _text_artifact("hello cache")
    r1 = fingerprint_artifact(artifact, config=config, services=services, ctx=ctx)
    assert not r1.cache_hit
    # corrupt the single stored object
    for key in list(cache.objects):
        cache.objects[key] = b'{"schema": 42, "garbage'
    r2 = fingerprint_artifact(artifact, config=config, services=services, ctx=ctx)
    assert not r2.cache_hit
    assert any(d.code == "CACHE_CORRUPT" for d in r2.warnings)
    assert scientific_sha256(r1.fingerprint) == scientific_sha256(r2.fingerprint)


def test_no_cache_flag() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config = StylogConfig()
    ctx = build_context(config, services)
    artifact = _text_artifact("hello")
    fingerprint_artifact(artifact, config=config, services=services, ctx=ctx, no_cache=True)
    assert cache.objects == {}


def test_filesystem_cache_roundtrip_and_layout(tmp_path) -> None:
    store = FilesystemCacheStore(tmp_path)
    cache = _services(store)
    config = StylogConfig()
    ctx = build_context(config, cache)
    artifact = _text_artifact("filesystem cache sample")
    r1 = fingerprint_artifact(artifact, config=config, services=cache, ctx=ctx)
    assert not r1.cache_hit
    r2 = fingerprint_artifact(artifact, config=config, services=cache, ctx=ctx)
    assert r2.cache_hit
    assert scientific_sha256(r1.fingerprint) == scientific_sha256(r2.fingerprint)
    objects = list((tmp_path / "objects").rglob("*.json"))
    assert len(objects) == 1
    # stored bytes are canonical + single trailing LF and validate strictly
    data = objects[0].read_bytes()
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    parsed = model_from_bytes(data, Fingerprint)
    assert parsed.artifact.artifact_id == "t"


def test_concurrent_writers_same_key(tmp_path) -> None:
    store = FilesystemCacheStore(tmp_path)
    payload = b'{"schema":"x"}'
    for _ in range(5):
        store.put("ab" + "0" * 62, payload)
    assert store.get("ab" + "0" * 62) == payload
