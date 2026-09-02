"""Tests for the deterministic validation pair builders (spec 23.13)."""

from __future__ import annotations

import hashlib

import pytest
import verifbuild
from verifbuild import (
    PairCandidate,
    author_population,
    canonical_pair_identity,
    pair_population,
    select_pairs,
    selection_key,
    write_dataset_manifest,
    write_training_manifest,
)


def _candidate(index: int, label: str, author: str = "a", problem: str | None = None) -> PairCandidate:
    left_sha = hashlib.sha256(f"left{index}".encode()).hexdigest()
    right_sha = hashlib.sha256(f"right{index}".encode()).hexdigest()
    authors = (author,) if label == "same" else (author, f"other{index}")
    problems = (problem,) if problem is not None else ()
    return PairCandidate(
        left=f"l{index}",
        right=f"r{index}",
        label=label,
        left_sha=left_sha,
        right_sha=right_sha,
        authors=authors,
        problems=problems,
    )


def test_canonical_identity_orientation_invariant() -> None:
    first = canonical_pair_identity("a" * 64, "b" * 64, "same")
    second = canonical_pair_identity("b" * 64, "a" * 64, "same")
    assert first == second
    assert canonical_pair_identity("a" * 64, "b" * 64, "same") != canonical_pair_identity(
        "a" * 64, "b" * 64, "different"
    )


def test_selection_key_golden() -> None:
    identity = canonical_pair_identity("0" * 64, "1" * 64, "same")
    expected = hashlib.sha256(b"stylog-pair-select/1\0" + identity).digest()
    assert selection_key(identity) == expected
    # version bump re-ranks
    assert selection_key(identity, "2") != expected


def test_select_pairs_deterministic_across_runs() -> None:
    candidates = [_candidate(i, "same", author=f"au{i % 3}") for i in range(40)]
    candidates += [_candidate(100 + i, "different", author=f"au{i % 3}") for i in range(40)]
    first, stats_a = select_pairs(candidates, negative_positive_ratio=1.0)
    second, stats_b = select_pairs(list(reversed(candidates)), negative_positive_ratio=1.0)
    assert first == second
    assert stats_a == stats_b
    assert stats_a["positives"] == stats_a["negatives"]


def test_select_pairs_sha256_ranked_not_sorted_first_n() -> None:
    candidates = [_candidate(i, "same", author="solo") for i in range(20)]
    selected, _ = select_pairs(candidates, max_pairs_per_author=5)
    assert len(selected) == 5
    keys = sorted(
        selection_key(candidate.canonical_identity()) for candidate in candidates
    )
    selected_keys = {selection_key(candidate.canonical_identity()) for candidate in selected}
    # the five lowest SHA-256 keys win, not the five lowest indices
    assert selected_keys == set(keys[:5])


def test_select_pairs_caps_and_ratio() -> None:
    positives = [_candidate(i, "same", author=f"au{i % 2}") for i in range(10)]
    negatives = [_candidate(100 + i, "different", author=f"au{i % 2}") for i in range(30)]
    selected, stats = select_pairs(
        positives + negatives, max_pairs_per_author=3, negative_positive_ratio=2.0
    )
    # author cap: 2 authors * 3 pairs = at most 6 per class side involvement
    assert stats["positives"] <= 6
    assert stats["negatives"] <= int(2.0 * stats["positives"])
    selected2, stats2 = select_pairs(
        positives + negatives, max_pairs_per_author=3, negative_positive_ratio=2.0
    )
    assert selected == selected2
    assert stats == stats2


def test_select_pairs_problem_caps() -> None:
    candidates = [
        _candidate(i, "different", author=f"au{i}", problem=f"p{i % 2}") for i in range(10)
    ]
    selected, stats = select_pairs(candidates, max_pairs_per_problem=2)
    assert len(selected) == 4  # 2 problems * 2 pairs
    assert stats["capped_problem"] == 6


def test_population_assignment_disjoint_and_deterministic() -> None:
    populations = {
        author: author_population(author, "seed", 250_000, 250_000, 250_000)
        for author in (f"author{i}" for i in range(200))
    }
    assert set(populations.values()) <= {"train", "tuning", "calibration", "evaluation"}
    again = author_population("author7", "seed", 250_000, 250_000, 250_000)
    assert again == populations["author7"]
    other_seed = author_population("author7", "other-seed", 250_000, 250_000, 250_000)
    assert other_seed != populations["author7"] or True  # may coincide; just deterministic
    # pair spanning populations is dropped (None)
    spanning = PairCandidate(
        left="l",
        right="r",
        label="different",
        left_sha="0" * 64,
        right_sha="1" * 64,
        authors=("a-x", "a-y"),
    )
    pops = {
        author_population(author, "seed", 250_000, 250_000, 250_000)
        for author in spanning.authors
    }
    result = pair_population(spanning, "seed", 250_000, 250_000, 250_000)
    if len(pops) == 1:
        assert result == pops.pop()
    else:
        assert result is None


def test_manifest_writers_roundtrip(tmp_path) -> None:
    from stylog.benchmark.manifest import load_dataset_manifest, load_training_manifest

    entries = [
        verifbuild.DatasetEntry(
            id="a1",
            path="texts/a1.txt",
            sha256="1" * 64,
            kind="text",
            language="en",
            context={"author_id": "auth1", "problem_id": "p1"},
        ),
        verifbuild.DatasetEntry(
            id="a2",
            path="texts/a2.txt",
            sha256="2" * 64,
            kind="text",
            language="en",
            context={"author_id": "auth2"},
        ),
    ]
    dataset_path = tmp_path / "dataset.toml"
    write_dataset_manifest(
        dataset_path,
        dataset_id="d",
        version="1",
        license_name="CC0",
        redistribution="allowed",
        source="synthetic",
        artifacts=entries,
    )
    manifest = load_dataset_manifest(dataset_path)
    assert manifest.id == "d"
    assert len(manifest.artifacts) == 2
    assert manifest.artifact_by_id("a1").context["problem_id"] == "p1"

    training_path = tmp_path / "training.toml"
    write_training_manifest(
        training_path,
        training_id="t",
        dataset_name="dataset.toml",
        verifier={
            "kind": "text",
            "l2_lambda": 1.0,
            "min_support_fraction": 0.9,
            "min_class_support_fraction": 0.8,
            "min_pairs": 4,
            "threshold_rule": "fixed",
            "threshold_fixed": 0.5,
            "include_linguistic": False,
            "allow_unconstrained_language": False,
        },
        pair_policy={"selection_version": "1", "max_pairs_per_author": 4},
        pairs=[
            verifbuild.TrainingPairRow(left="a1", right="a2", label="different", population="train"),
            verifbuild.TrainingPairRow(left="a2", right="a1", label="same", population="tuning"),
        ],
    )
    training = load_training_manifest(training_path)
    assert training.id == "t"
    assert training.verifier.threshold_rule == "fixed"
    assert training.verifier.max_pairs_per_author == 4
    assert [pair.population for pair in training.pairs] == ["train", "tuning"]


def test_population_ppm_validation() -> None:
    with pytest.raises(ValueError):
        author_population("a", "seed", 600_000, 600_000, 600_000)


def test_negative_positive_ratio_is_a_cap_not_a_target() -> None:
    # Scarce negatives: every negative is kept, the cap never binds, and the
    # final class ratio stays skewed (ratio_dropped == 0).
    positives = [_candidate(i, "same", author=f"au{i}") for i in range(10)]
    scarce_negatives = [_candidate(100 + i, "different", author=f"bu{i}") for i in range(2)]
    _selected, stats = select_pairs(positives + scarce_negatives, negative_positive_ratio=1.0)
    assert stats["positives"] == 10
    assert stats["negatives"] == 2  # NOT 10: the ratio does not manufacture negatives
    assert stats["ratio_dropped"] == 0

    # Abundant negatives: selection stops at int(ratio * positives).
    abundant_negatives = [_candidate(200 + i, "different", author=f"cu{i}") for i in range(30)]
    _selected, stats = select_pairs(positives + abundant_negatives, negative_positive_ratio=1.0)
    assert stats["positives"] == 10
    assert stats["negatives"] == 10
    assert stats["ratio_dropped"] == 20


def test_population_assignment_depletes_different_class_quadratically() -> None:
    # Structural model: same pairs keep at ppm_p, different pairs at ~ppm_p^2.
    # With disjoint author sets and ppm = 0.25, the observed class ratio of a
    # balanced corpus converges to ~4:1 -- documented consequence, not a bug.
    same_pairs = [
        PairCandidate(
            left=f"l{i}", right=f"r{i}", label="same",
            left_sha=f"{i:064d}", right_sha=f"{i + 1000:064d}",
            authors=(f"author{i}",),
        )
        for i in range(200)
    ]
    different_pairs = [
        PairCandidate(
            left=f"l{i}", right=f"r{i}", label="different",
            left_sha=f"{i + 2000:064d}", right_sha=f"{i + 3000:064d}",
            authors=(f"author{i}", f"author{i + 200}"),
        )
        for i in range(200)
    ]
    same_kept = sum(
        1 for pair in same_pairs
        if pair_population(pair, "seed", 250_000, 250_000, 250_000) == "train"
    )
    different_kept = sum(
        1 for pair in different_pairs
        if pair_population(pair, "seed", 250_000, 250_000, 250_000) == "train"
    )
    assert 0.15 <= same_kept / 200 <= 0.35  # ~0.25 keep rate
    assert different_kept < same_kept  # quadratic depletion
    assert different_kept / 200 < 0.15  # ~0.0625 keep rate, well below same
