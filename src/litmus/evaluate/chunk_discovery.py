"""Alternative chunk set discovery (medium+ tier).

After scoring a RAG answer: if it scores high (>0.8 overall) but the chunks
it actually retrieved don't match any annotated gold chunk set, the
retrieved set represents a valid alternative path the generator missed.
Auto-add it as a discovered alternative rather than penalizing a correct
retrieval.
"""

from __future__ import annotations

from litmus.config import CHUNK_DISCOVERY_SCORE_THRESHOLD
from litmus.evaluate.retrieval_scorer import chunk_is_retrieved
from litmus.models import EvalRecord


def discover_alternative_chunks(
    record: EvalRecord,
    retrieved_contexts: list[str],
    overall_score: float,
    set_recall_hit: bool,
) -> list[str] | None:
    """Return newly discovered gold-worthy contexts, or None if nothing to add."""
    if set_recall_hit or overall_score < CHUNK_DISCOVERY_SCORE_THRESHOLD or record.unanswerable:
        return None

    all_gold_texts = {text for group in record.gold_chunks_text for text in group}
    novel_contexts = [
        ctx
        for ctx in retrieved_contexts
        if ctx.strip() and not any(chunk_is_retrieved(gold_text, [ctx]) for gold_text in all_gold_texts)
    ]
    return novel_contexts or None


def apply_discovered_alternatives(record: EvalRecord, discovered_contexts: list[str]) -> None:
    """Mutate the record in place, adding a new gold_chunk_ids/gold_chunks_text alternative.

    Since discovered contexts are free text (not resolved chunk IDs - the
    RAG system may chunk differently than litmus did), the new alternative
    group uses synthetic placeholder IDs so gold_chunk_ids and
    gold_chunks_text stay index-aligned.
    """
    if not discovered_contexts:
        return
    placeholder_ids = [f"discovered#{record.id}#{i}" for i in range(len(discovered_contexts))]
    record.gold_chunk_ids.append(placeholder_ids)
    record.gold_chunks_text.append(discovered_contexts)
