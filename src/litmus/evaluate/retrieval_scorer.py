"""Context matching and retrieval metrics: set recall, chunk recall, MRR,
precision@K, gold chunk rank positions.

The RAG callable returns free-text ``contexts``, not chunk IDs, so matching
retrieved contexts back to annotated gold chunks is a text-matching
problem. We use a two-stage match: exact (normalized) substring containment
first, then a token-overlap (Jaccard) fallback for near-identical text that
differs in whitespace/punctuation from re-chunking. This stays fast and
deterministic (no embedding calls in the hot evaluation loop) and is easy
to unit test without an LLM or network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chunk_is_retrieved(gold_chunk_text: str, retrieved_contexts: list[str], jaccard_threshold: float = 0.5) -> bool:
    """True if a gold chunk's text appears (verbatim or near-duplicate) among retrieved contexts."""
    if not gold_chunk_text.strip():
        return False
    normalized_gold = _normalize(gold_chunk_text)
    gold_tokens = _tokens(gold_chunk_text)
    for ctx in retrieved_contexts:
        normalized_ctx = _normalize(ctx)
        if normalized_gold in normalized_ctx or normalized_ctx in normalized_gold:
            return True
        if _jaccard(gold_tokens, _tokens(ctx)) >= jaccard_threshold:
            return True
    return False


def first_retrieval_rank(gold_chunk_text: str, retrieved_contexts: list[str], jaccard_threshold: float = 0.5) -> int | None:
    """1-indexed rank of the first retrieved context matching this gold chunk, or None."""
    normalized_gold = _normalize(gold_chunk_text)
    gold_tokens = _tokens(gold_chunk_text)
    for rank, ctx in enumerate(retrieved_contexts, start=1):
        normalized_ctx = _normalize(ctx)
        if normalized_gold in normalized_ctx or normalized_ctx in normalized_gold:
            return rank
        if _jaccard(gold_tokens, _tokens(ctx)) >= jaccard_threshold:
            return rank
    return None


@dataclass
class RetrievalScore:
    chunk_recall: float
    set_recall: bool
    mrr: float | None
    precision_at_k: float | None
    gold_chunk_ranks: list[int] | None


def score_retrieval(
    gold_chunks_text: list[list[str]],
    retrieved_contexts: list[str],
    compute_mrr: bool = False,
    compute_precision: bool = False,
    compute_ranks: bool = False,
) -> RetrievalScore:
    """Score one record's retrieval against its alternative gold chunk sets.

    gold_chunks_text: list of alternative valid chunk-text groups (any one
    complete group being retrieved counts as a set-recall hit).
    """
    all_gold_texts = [text for group in gold_chunks_text for text in group]

    if not all_gold_texts:
        # Unanswerable question: there's nothing to retrieve, so recall
        # metrics are vacuously perfect and not meaningful to report.
        return RetrievalScore(chunk_recall=1.0, set_recall=True, mrr=None, precision_at_k=None, gold_chunk_ranks=None)

    retrieved_flags = [chunk_is_retrieved(text, retrieved_contexts) for text in all_gold_texts]
    chunk_recall = sum(retrieved_flags) / len(all_gold_texts)

    set_recall = any(
        all(chunk_is_retrieved(text, retrieved_contexts) for text in group) for group in gold_chunks_text if group
    )

    mrr = None
    if compute_mrr:
        ranks = [first_retrieval_rank(text, retrieved_contexts) for text in all_gold_texts]
        valid_ranks = [r for r in ranks if r is not None]
        mrr = (sum(1.0 / r for r in valid_ranks) / len(valid_ranks)) if valid_ranks else 0.0

    precision_at_k = None
    if compute_precision and retrieved_contexts:
        relevant = sum(
            1
            for ctx in retrieved_contexts
            if any(
                _normalize(text) in _normalize(ctx)
                or _normalize(ctx) in _normalize(text)
                or _jaccard(_tokens(text), _tokens(ctx)) >= 0.5
                for text in all_gold_texts
            )
        )
        precision_at_k = relevant / len(retrieved_contexts)

    gold_chunk_ranks = None
    if compute_ranks:
        gold_chunk_ranks = [r for r in (first_retrieval_rank(text, retrieved_contexts) for text in all_gold_texts) if r is not None]

    return RetrievalScore(
        chunk_recall=chunk_recall,
        set_recall=set_recall,
        mrr=mrr,
        precision_at_k=precision_at_k,
        gold_chunk_ranks=gold_chunk_ranks,
    )
