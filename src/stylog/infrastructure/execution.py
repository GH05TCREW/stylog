"""Serial/process execution with deterministic ordered results (spec 18.16).

Ordinals are assigned before dispatch; results are emitted in ordinal order
regardless of completion order; in-flight work is bounded. Worker count and
scheduling never enter scientific artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_ordered(
    items: Iterable[T],
    fn: Callable[[T], R],
    *,
    mode: str = "serial",
    workers: int = 0,
    max_in_flight: int = 0,
) -> Iterator[R]:
    """Yield fn(item) for each item in input order.

    ``fn`` must be a module-level picklable callable when mode="process".
    """
    indexed = list(enumerate(items))
    if mode == "serial" or workers == 1 or len(indexed) <= 1:
        for _, item in indexed:
            yield fn(item)
        return

    pool_workers = workers if workers > 0 else None
    bound = max_in_flight if max_in_flight > 0 else (pool_workers or 4) * 4
    results: dict[int, R] = {}
    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        pending = set()
        iterator = iter(indexed)

        def submit_up_to_bound() -> None:
            nonlocal pending
            while len(pending) < bound:
                try:
                    ordinal, item = next(iterator)
                except StopIteration:
                    return
                future = pool.submit(fn, item)
                future._stylog_ordinal = ordinal  # type: ignore[attr-defined]
                pending.add(future)

        submit_up_to_bound()
        from concurrent.futures import as_completed

        next_emit = 0
        pending_futures = pending
        while pending_futures:
            for future in as_completed(list(pending_futures)):
                ordinal = future._stylog_ordinal  # type: ignore[attr-defined]
                results[ordinal] = future.result()
                pending_futures.discard(future)
                submit_up_to_bound()
                break
            while next_emit in results:
                yield results.pop(next_emit)
                next_emit += 1
