"""Regression test: cache hits must not leak a foreign artifact_id (DEFECT-1).

Two artifacts with identical content share a cache key (content+config
identity), but each artifact instance MUST keep its own artifact_id
(spec 18.1). A cache hit rewrites the descriptor's artifact_id to the
requesting artifact.
"""

from __future__ import annotations

from stylog.application.batch import analyze_iter
from stylog.application.fingerprint import fingerprint_artifact
from stylog.bootstrap import build_context
from stylog.config import StylogConfig
from stylog.infrastructure.baselines import StaticBaselineResolver
from stylog.infrastructure.cache import MemoryCacheStore
from stylog.infrastructure.ingest import artifact_from_text
from stylog.infrastructure.resources import PackageResourceResolver
from stylog.ports import RuntimeServices


def _services(cache) -> RuntimeServices:
    return RuntimeServices(
        cache=cache, baselines=StaticBaselineResolver({}), resources=PackageResourceResolver()
    )


def test_cache_hit_rewrites_instance_id() -> None:
    cache = MemoryCacheStore()
    services = _services(cache)
    config = StylogConfig()
    ctx = build_context(config, services)

    first = artifact_from_text("identical content here", artifact_id="instance-a")
    second = artifact_from_text("identical content here", artifact_id="instance-b")

    r1 = fingerprint_artifact(first, config=config, services=services, ctx=ctx)
    assert not r1.cache_hit
    r2 = fingerprint_artifact(second, config=config, services=services, ctx=ctx)
    assert r2.cache_hit
    assert r2.fingerprint.artifact.artifact_id == "instance-b"
    assert r1.fingerprint.artifact.artifact_id == "instance-a"
    # scientific content is identical apart from the instance id
    assert (
        r1.fingerprint.artifact.content_identity
        == r2.fingerprint.artifact.content_identity
    )


def test_batch_duplicate_content_keeps_all_instances(tmp_path, monkeypatch) -> None:
    config = StylogConfig()
    artifacts = [
        artifact_from_text("same text body", artifact_id="a"),
        artifact_from_text("same text body", artifact_id="b"),
        artifact_from_text("same text body", artifact_id="c"),
    ]
    monkeypatch.setenv("STYLOG_CACHE_DIR", str(tmp_path / "cache"))
    bundles = list(analyze_iter(artifacts, config=config, execution="process", workers=2))
    ids = sorted(bundle.primary.artifact.artifact_id for bundle, _ in bundles)
    assert ids == ["a", "b", "c"]
