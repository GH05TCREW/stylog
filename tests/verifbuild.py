"""Deterministic pair construction for verification manifests (spec 23.13).

Shared, stdlib-only library for the validation builders. Everything here is
exactly reproducible: candidate pairs are enumerated in a fully specified
sorted order, selection within a class stratum uses SHA-256 ranking (never
sorted-first-N, so author ids / problem ids / filenames / chronology cannot
bias the selection), and population assignment uses SHA-256 hash buckets over
author ids. No RNG anywhere.

Pipeline for a builder:
1. enumerate ``PairCandidate`` records (sorted by canonical pair identity),
2. drop exact duplicate artifacts BEFORE population assignment (the
   duplicate-content guard: canonical pair identity over sorted content
   hashes; keep the smallest artifact id),
3. assign each pair to the population shared by its authors (pairs whose
   authors span populations are dropped, counted),
4. ``select_pairs`` per population under the pair policy,
5. emit dataset/training manifests with ``write_*`` helpers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SELECTION_VERSION = "1"
POPULATIONS = ("train", "tuning", "calibration", "evaluation")
_SPLIT_ALGORITHM_ID = "stylog-verif-split-v1"
_PPM_TOTAL = 1_000_000


@dataclass(frozen=True)
class PairCandidate:
    """One labeled candidate pair before selection."""

    left: str  # artifact id (canonically the smaller content sha side is NOT
    right: str  # required; refs stay as constructed)
    label: str  # "same" | "different"
    left_sha: str  # content sha256 of the left artifact
    right_sha: str
    authors: tuple[str, ...]  # 1 for same-author, 2 for different-author
    problems: tuple[str, ...] = ()  # problem ids for code cells; empty for text

    def canonical_identity(self) -> bytes:
        """sha256_left ‖ sha256_right ‖ label in canonically ordered form."""
        first, second = sorted((self.left_sha, self.right_sha))
        return canonical_pair_identity(first, second, self.label)


def canonical_pair_identity(left_sha: str, right_sha: str, label: str) -> bytes:
    """Canonical pair identity: ordered content hashes plus label (spec 23.13)."""
    first, second = sorted((left_sha, right_sha))
    return f"{first}\x00{second}\x00{label}".encode()


def selection_key(identity: bytes, selection_version: str = SELECTION_VERSION) -> bytes:
    """Deterministic SHA-256 ranking key for one candidate pair."""
    return hashlib.sha256(
        b"stylog-pair-select/" + selection_version.encode("utf-8") + b"\0" + identity
    ).digest()


def author_population(
    author_id: str,
    seed: str,
    train_ppm: int,
    tuning_ppm: int,
    calibration_ppm: int,
) -> str:
    """Author-disjoint population assignment via SHA-256 hash buckets.

    bucket = SHA256("stylog-verif-split-v1\\0" + seed + "\\0" + author_id) mod 1e6;
    [0, train) -> train, [train, +tuning) -> tuning, [.., +calibration) ->
    calibration, remainder -> evaluation.
    """
    if train_ppm + tuning_ppm + calibration_ppm > _PPM_TOTAL:
        raise ValueError("population ppm values exceed 1_000_000")
    payload = f"{_SPLIT_ALGORITHM_ID}\0{seed}\0{author_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest(), "big") % _PPM_TOTAL
    if bucket < train_ppm:
        return "train"
    if bucket < train_ppm + tuning_ppm:
        return "tuning"
    if bucket < train_ppm + tuning_ppm + calibration_ppm:
        return "calibration"
    return "evaluation"


def pair_population(
    candidate: PairCandidate,
    seed: str,
    train_ppm: int,
    tuning_ppm: int,
    calibration_ppm: int,
) -> str | None:
    """Population shared by all of the pair's authors; None when spanning."""
    populations = {
        author_population(author, seed, train_ppm, tuning_ppm, calibration_ppm)
        for author in candidate.authors
    }
    if len(populations) != 1:
        return None
    return populations.pop()


@dataclass
class SelectionStats:
    offered: int = 0
    selected: int = 0
    capped_author: int = 0
    capped_problem: int = 0
    ratio_dropped: int = 0


def _accepts(
    candidate: PairCandidate,
    author_counts: dict[str, int],
    problem_counts: dict[str, int],
    max_pairs_per_author: int | None,
    max_pairs_per_problem: int | None,
) -> str | None:
    if max_pairs_per_author is not None:
        for author in candidate.authors:
            if author_counts.get(author, 0) >= max_pairs_per_author:
                return "author"
    if max_pairs_per_problem is not None:
        for problem in candidate.problems:
            if problem_counts.get(problem, 0) >= max_pairs_per_problem:
                return "problem"
    return None


def select_pairs(
    candidates: Iterable[PairCandidate],
    *,
    max_pairs_per_author: int | None = None,
    max_pairs_per_problem: int | None = None,
    negative_positive_ratio: float | None = None,
    selection_version: str = SELECTION_VERSION,
) -> tuple[list[PairCandidate], dict[str, int]]:
    """SHA-256-ranked selection with deterministic caps and class ratio.

    Positives (label "same") are selected first in ascending selection-key
    order under the caps; negatives are then selected in ascending key order
    under the caps up to ``negative_positive_ratio * len(positives)`` when a
    ratio is declared. Deterministic for a fixed selection_version.

    Semantics of ``negative_positive_ratio`` (normative, unambiguous): it is
    a CAP on the negative count, never a TARGET. It does not manufacture
    negatives — when the negative pool is smaller than the budget, every
    negative is kept and the final class ratio stays skewed
    (``ratio_dropped == 0``). It is applied per population AFTER
    author-disjoint population assignment.

    Note on population assignment (see ``pair_population``): same-author
    pairs carry one author and are assigned with probability ppm_p, while
    different-author pairs are assigned only when BOTH authors land in
    population p (probability ~ ppm_p^2). Author-disjoint assignment
    therefore depletes the different class roughly quadratically relative to
    the same class even when the source corpus is perfectly balanced — the
    final class ratio per population is a structural consequence of the ppm
    schedule, not of this function.
    """
    ranked = sorted(
        candidates, key=lambda candidate: selection_key(candidate.canonical_identity(), selection_version)
    )
    positives = [candidate for candidate in ranked if candidate.label == "same"]
    negatives = [candidate for candidate in ranked if candidate.label != "same"]
    stats = SelectionStats(offered=len(ranked))
    author_counts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}
    selected: list[PairCandidate] = []

    def take(candidate: PairCandidate) -> bool:
        blocked = _accepts(
            candidate,
            author_counts,
            problem_counts,
            max_pairs_per_author,
            max_pairs_per_problem,
        )
        if blocked == "author":
            stats.capped_author += 1
            return False
        if blocked == "problem":
            stats.capped_problem += 1
            return False
        for author in candidate.authors:
            author_counts[author] = author_counts.get(author, 0) + 1
        for problem in candidate.problems:
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
        selected.append(candidate)
        return True

    n_positive = 0
    for candidate in positives:
        if take(candidate):
            n_positive += 1
    if negative_positive_ratio is None:
        negative_budget: int | None = None
    else:
        negative_budget = int(negative_positive_ratio * n_positive)
    n_negative = 0
    for candidate in negatives:
        if negative_budget is not None and n_negative >= negative_budget:
            stats.ratio_dropped += 1
            continue
        if take(candidate):
            n_negative += 1
    stats.selected = len(selected)
    # Final output order: ascending selection key (fully specified).
    selected.sort(key=lambda candidate: selection_key(candidate.canonical_identity(), selection_version))
    return selected, {
        "offered": stats.offered,
        "selected": stats.selected,
        "capped_author": stats.capped_author,
        "capped_problem": stats.capped_problem,
        "ratio_dropped": stats.ratio_dropped,
        "positives": n_positive,
        "negatives": n_negative,
    }


# ---------------------------------------------------------------------------
# Manifest emission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    path: str  # relative to the dataset root
    sha256: str
    kind: str
    language: str
    context: dict[str, str] = field(default_factory=dict)


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def write_dataset_manifest(
    path: Path,
    *,
    dataset_id: str,
    version: str,
    license_name: str,
    redistribution: str,
    source: str,
    artifacts: Iterable[DatasetEntry],
) -> None:
    lines = [
        'schema = "stylog.dataset"',
        'schema_version = "0.1.0"',
        f"id = {_toml_str(dataset_id)}",
        f"version = {_toml_str(version)}",
        f"license = {_toml_str(license_name)}",
        f"redistribution = {_toml_str(redistribution)}",
        f"source = {_toml_str(source)}",
        "",
    ]
    for artifact in artifacts:
        lines.append("[[artifact]]")
        lines.append(f"id = {_toml_str(artifact.id)}")
        lines.append(f"path = {_toml_str(artifact.path)}")
        lines.append(f"sha256 = {_toml_str(artifact.sha256)}")
        lines.append(f"kind = {_toml_str(artifact.kind)}")
        lines.append(f"language = {_toml_str(artifact.language)}")
        for key in sorted(artifact.context):
            lines.append(f"{key} = {_toml_str(artifact.context[key])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class TrainingPairRow:
    left: str
    right: str
    label: str
    population: str  # train | tuning | calibration


def write_training_manifest(
    path: Path,
    *,
    training_id: str,
    dataset_name: str,
    verifier: dict[str, object],
    pair_policy: dict[str, object],
    pairs: Iterable[TrainingPairRow],
) -> None:
    lines = [
        'schema = "stylog.verifier-training"',
        'schema_version = "0.1.0"',
        f"id = {_toml_str(training_id)}",
        f"dataset = {_toml_str(dataset_name)}",
        "",
        "[verifier]",
    ]
    for key, value in verifier.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value!r}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{key} = [{', '.join(_toml_str(item) for item in value)}]")
        else:
            lines.append(f"{key} = {_toml_str(str(value))}")
    lines.append("")
    lines.append("[verifier.pair_policy]")
    for key, value in pair_policy.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value!r}")
        else:
            lines.append(f"{key} = {_toml_str(str(value))}")
    lines.append("")
    for pair in pairs:
        lines.append("[[pair]]")
        lines.append(f"left = {_toml_str(pair.left)}")
        lines.append(f"right = {_toml_str(pair.right)}")
        lines.append(f"label = {_toml_str(pair.label)}")
        lines.append(f"population = {_toml_str(pair.population)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_evaluation_benchmark_spec(
    path: Path,
    *,
    benchmark_id: str,
    dataset_name: str,
    model_name: str,
    pairs: Iterable[tuple[str, str, str]],
) -> None:
    """A stylog.benchmark verification spec for the held-out population."""
    lines = [
        'schema = "stylog.benchmark"',
        'schema_version = "0.1.0"',
        f"id = {_toml_str(benchmark_id)}",
        'task = "verification"',
        f"dataset = {_toml_str(dataset_name)}",
        "",
        "[verifier]",
        f"model = {_toml_str(model_name)}",
        "",
    ]
    for left, right, label in pairs:
        lines.append("[[pair]]")
        lines.append(f"left = {_toml_str(left)}")
        lines.append(f"right = {_toml_str(right)}")
        lines.append(f"label = {_toml_str(label)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
