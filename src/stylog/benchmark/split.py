"""Deterministic dataset splitting (spec section 21).

Artifacts are unioned into components by shared ``disjoint_by`` context
values; each component is assigned wholesale to train/dev/test by a
SHA-256 bucket derived from the seed and the component key. No ``random``
module, no fallback when a split is impossible or leaky.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from stylog.benchmark.manifest import DatasetManifest, SplitSpec, _fail
from stylog.serialization.canonical import sha256_of_tree

SPLIT_ALGORITHM_VERSION = "stylog-split-v1"
_PPM_TOTAL = 1_000_000


@dataclass(frozen=True)
class SplitResult:
    assignment: dict[str, str]  # artifact id -> "train" | "dev" | "test"
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    split_config_sha256: str
    algorithm_version: str


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self._parent = {artifact_id: artifact_id for artifact_id in ids}

    def find(self, artifact_id: str) -> str:
        root = artifact_id
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[artifact_id] != root:  # path compression
            self._parent[artifact_id], artifact_id = root, self._parent[artifact_id]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Deterministic: the lexicographically smaller root wins.
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            self._parent[right_root] = left_root


def _bucket(seed: str, component_key: str) -> int:
    payload = f"{SPLIT_ALGORITHM_VERSION}\0{seed}\0{component_key}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") % _PPM_TOTAL


def compute_split(manifest: DatasetManifest, split: SplitSpec) -> SplitResult:
    """Realize the split; raise SPLIT_IMPOSSIBLE/SPLIT_LEAKAGE on violation."""
    ids = sorted(artifact.id for artifact in manifest.artifacts)
    union_find = _UnionFind(ids)
    for field_name in split.disjoint_by:
        groups: dict[str, list[str]] = {}
        for artifact in manifest.artifacts:
            value = artifact.context.get(field_name, "")
            if value == "":
                if split.require_nonempty:
                    _fail(
                        "SPLIT_IMPOSSIBLE",
                        f"artifact {artifact.id!r} has no value for disjoint_by field "
                        f"{field_name!r} while require_nonempty is true",
                    )
                continue
            groups.setdefault(value, []).append(artifact.id)
        for group in groups.values():
            for other in group[1:]:
                union_find.union(group[0], other)

    if split.disjoint_content:
        # Duplicate-content guard: artifacts with identical content hashes
        # must never land in different split parts (default off).
        by_hash: dict[str, list[str]] = {}
        for artifact in manifest.artifacts:
            by_hash.setdefault(artifact.sha256, []).append(artifact.id)
        for group in by_hash.values():
            for other in group[1:]:
                union_find.union(group[0], other)

    components: dict[str, list[str]] = {}
    for artifact_id in ids:
        components.setdefault(union_find.find(artifact_id), []).append(artifact_id)

    assignment: dict[str, str] = {}
    for members in components.values():
        component_key = min(members)
        bucket = _bucket(split.seed, component_key)
        if bucket < split.train_ppm:
            part = "train"
        elif bucket < split.train_ppm + split.dev_ppm:
            part = "dev"
        else:
            part = "test"
        for artifact_id in members:
            assignment[artifact_id] = part

    realized: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for artifact_id in ids:
        realized[assignment[artifact_id]].append(artifact_id)

    if split.require_nonempty:
        for part, ppm in (
            ("train", split.train_ppm),
            ("dev", split.dev_ppm),
            ("test", split.test_ppm),
        ):
            if ppm > 0 and not realized[part]:
                _fail(
                    "SPLIT_IMPOSSIBLE",
                    f"split part {part!r} has ppm {ppm} but received zero components",
                )

    # Post-check: no disjoint_by value may appear in two split parts.
    for field_name in split.disjoint_by:
        seen: dict[str, str] = {}
        for artifact in manifest.artifacts:
            value = artifact.context.get(field_name, "")
            if value == "":
                continue
            part = assignment[artifact.id]
            previous = seen.setdefault(value, part)
            if previous != part:
                _fail(
                    "SPLIT_LEAKAGE",
                    f"disjoint_by value {value!r} of field {field_name!r} appears in "
                    f"both {previous!r} and {part!r}",
                )

    return SplitResult(
        assignment=assignment,
        train=tuple(realized["train"]),
        dev=tuple(realized["dev"]),
        test=tuple(realized["test"]),
        split_config_sha256=sha256_of_tree(split.as_tree()),
        algorithm_version=SPLIT_ALGORITHM_VERSION,
    )
