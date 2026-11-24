"""Shared, reviewer-friendly sampling policy for verification workflows."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from typing import TypeVar


T = TypeVar("T")
STRATEGY_QUOTAS = (("Pulse", 1), ("NT", 1), ("CT", 1), ("AT50", 2), ("AT100", 2), ("Dens", 2), ("ATAgent2", 2))
SAMPLE_STRATEGY_RE = re.compile(r"^[A-H]\d{1,2}_(?P<strategy>.+)-[A-H]\d{1,2}$", re.IGNORECASE)


def strategy_from_sample(sample: str) -> str:
    match = SAMPLE_STRATEGY_RE.match(str(sample))
    return match.group("strategy") if match else str(sample)


def balanced_strategy_sample(candidates: Iterable[T], strategy_of: Callable[[T], str], seed: int) -> list[T]:
    """Select the fixed review cohort, skipping unavailable strategies."""
    groups: dict[str, list[T]] = {}
    for candidate in candidates:
        strategy = str(strategy_of(candidate))
        if strategy == "ATAgent1":
            continue
        groups.setdefault(strategy, []).append(candidate)
    rng = random.Random(seed)
    selected: list[T] = []
    for strategy, quota in STRATEGY_QUOTAS:
        pool = sorted(groups.get(strategy, []), key=str)
        selected.extend(rng.sample(pool, min(quota, len(pool))))
    return selected
