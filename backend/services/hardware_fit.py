"""Shared hardware-fit scoring primitives.

Both the text and image recommenders model hardware fit as the weighted sum
of three sub-scores (memory, speed, storage). The cost models differ — text
is memory-bandwidth-bound (TPS = bandwidth / active_size), image is
compute-bound (time = steps / TFLOPS) — and so do the memory thresholds
(diffusion is more punishing when overcommitted because runtimes hang
rather than degrade gracefully). This module collects the parts that ARE
shared so divergence stays intentional rather than accidental.
"""

from __future__ import annotations


# Combine weights for memory / speed / storage sub-scores. Both tracks use
# the same weighting because the underlying tradeoff is the same: must fit,
# must be usable, downloadable.
W_MEMORY = 0.50
W_SPEED = 0.40
W_STORAGE = 0.10


def combine_subscores(memory: int | float, speed: int | float, storage: int | float) -> float:
    """Weighted combine of the three sub-scores into a 0..10 hardware-fit score."""
    return round(W_MEMORY * memory + W_SPEED * speed + W_STORAGE * storage, 2)


# Storage thresholds — both tracks use the same ramp because the cost of a
# failed download is the same regardless of model type.
def score_storage_fit(download_gb: float, free_gb: float) -> int:
    """0..10 based on what fraction of free disk the download would consume."""
    ratio = download_gb / max(free_gb, 0.5)
    if ratio <= 0.05:
        return 10
    if ratio <= 0.15:
        return 8
    if ratio <= 0.30:
        return 6
    if ratio <= 0.60:
        return 4
    return 2


def bucket(value: float, thresholds: list[tuple[float, int]], default: int = 0) -> int:
    """Generic 'value <= threshold → score' lookup.

    `thresholds` is a list of (max_value_inclusive, score) pairs ordered by
    ascending max_value. Returns the score for the first threshold whose
    max_value covers `value`, else `default`.

    Example:
        bucket(mem_ratio, [(0.40, 10), (0.60, 9), (0.75, 8), ...])
    """
    for cap, score in thresholds:
        if value <= cap:
            return score
    return default
