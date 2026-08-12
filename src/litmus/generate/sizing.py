"""Corpus-aware eval-set sizing (Section 7 of the synth methodology doc).

`calculate_size` maps (num_docs, tier) to a question-count range using the
SIZING_TABLE lookup in config.py, then narrows to a point estimate using
num_chunks as a secondary signal (more chunks per doc within the same
bracket nudges toward the top of the range).
"""

from __future__ import annotations

from litmus.config import DOCUMENT_COVERAGE_TARGET, SIZING_TABLE


def _bracket_for(num_docs: int) -> dict[str, tuple[int, int]]:
    for upper_bound, ranges in SIZING_TABLE:
        if upper_bound is None or num_docs <= upper_bound:
            return ranges
    # Unreachable given the final (None, ...) sentinel, but keeps mypy happy.
    return SIZING_TABLE[-1][1]


def calculate_size(num_docs: int, num_chunks: int, tier: str) -> int:
    """Return a single point estimate for eval-set size, per Section 7 rules."""
    if num_docs <= 0:
        raise ValueError("num_docs must be positive")
    ranges = _bracket_for(num_docs)
    if tier not in ranges:
        raise ValueError(f"Unknown tier {tier!r}")
    low, high = ranges[tier]
    if low == high:
        return low

    # Chunks-per-doc as a density signal: denser corpora (more content per
    # doc) push toward the top of the range, sparse corpora toward the
    # bottom. avg_chunks_per_doc of ~20 is treated as "typical" (based on
    # the worked example: 1000 docs x 10pp x ~3 chunks/page = 30 chunks/doc,
    # scaled down since not every corpus is 10 pages/doc).
    avg_chunks_per_doc = num_chunks / num_docs if num_docs else 0
    density = min(max(avg_chunks_per_doc / 20.0, 0.0), 1.0)
    return round(low + density * (high - low))


def document_coverage_target(tier: str) -> tuple[float, float] | None:
    return DOCUMENT_COVERAGE_TARGET.get(tier)


def depth_for_document(num_chunks_in_doc: int) -> tuple[int, int]:
    """Per-document question-count range by complexity (Section 7.2 Layer 3)."""
    if num_chunks_in_doc <= 3:
        return (1, 1)
    if num_chunks_in_doc <= 10:
        return (2, 3)
    return (3, 5)
