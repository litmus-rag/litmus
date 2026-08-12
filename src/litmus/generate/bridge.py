"""Bridge-entity detection for cross-document question generation.

HopWeaver-style approach (Shen et al., 2025): find entities that appear in
chunks from more than one document, then use those as the anchor for
cross-doc question generation rather than naively picking two random
chunks. This avoids "pseudo multi-hop" questions that are secretly
answerable from a single document.

Entity extraction here is a lightweight LLM call per chunk (batched), not a
full NER pipeline — good enough to find plausible bridge candidates without
adding a heavy dependency.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from litmus.llm.client import LLMClient
from litmus.models import Chunk

_ENTITY_EXTRACTION_PROMPT = """Extract up to 5 distinctive named entities or key topics (product names, \
plan tiers, policy names, proper nouns, or specific technical terms) from this passage. Prefer specific \
terms over generic ones.

<passage>
{chunk_text}
</passage>

Output as a JSON array of strings, nothing else."""


@dataclass
class BridgeCandidate:
    entity: str
    chunk_ids: list[str]
    doc_ids: list[str]


def _normalize_entity(entity: str) -> str:
    return re.sub(r"\s+", " ", entity.strip().lower())


def extract_chunk_entities(chunk: Chunk, client: LLMClient) -> list[str]:
    prompt = _ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk.text[:2000])
    try:
        result = client.complete_json_array(prompt, temperature=0.0, max_tokens=200)
    except Exception:  # noqa: BLE001
        return []
    return [str(e) for e in result if str(e).strip()]


def _stratified_sample(chunks: dict[str, Chunk], limit: int) -> list[Chunk]:
    """Round-robin across documents so the scan covers many docs shallowly
    rather than few docs deeply. A plain prefix of ``chunks.values()`` is
    biased toward whichever documents were chunked first (chunk_documents()
    inserts chunk-by-chunk, doc-by-doc), which starves entity co-occurrence
    across documents of any chance to be observed."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks.values():
        by_doc[chunk.doc_id].append(chunk)

    sampled: list[Chunk] = []
    doc_queues = list(by_doc.values())
    round_idx = 0
    while len(sampled) < limit and any(round_idx < len(q) for q in doc_queues):
        for queue in doc_queues:
            if round_idx < len(queue):
                sampled.append(queue[round_idx])
                if len(sampled) >= limit:
                    break
        round_idx += 1
    return sampled


def find_bridge_candidates(
    chunks: dict[str, Chunk],
    client: LLMClient,
    min_docs: int = 2,
    max_chunks_to_scan: int | None = None,
) -> list[BridgeCandidate]:
    """Find entities that appear in chunks spanning at least `min_docs` documents."""
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    entity_display: dict[str, str] = {}

    items = _stratified_sample(chunks, max_chunks_to_scan) if max_chunks_to_scan is not None else list(chunks.values())

    for chunk in items:
        for entity in extract_chunk_entities(chunk, client):
            key = _normalize_entity(entity)
            if not key:
                continue
            entity_to_chunks[key].add(chunk.id)
            entity_display.setdefault(key, entity)

    candidates: list[BridgeCandidate] = []
    for key, chunk_ids in entity_to_chunks.items():
        doc_ids = {chunks[cid].doc_id for cid in chunk_ids}
        if len(doc_ids) >= min_docs:
            candidates.append(
                BridgeCandidate(
                    entity=entity_display[key],
                    chunk_ids=sorted(chunk_ids),
                    doc_ids=sorted(doc_ids),
                )
            )
    candidates.sort(key=lambda c: len(c.doc_ids), reverse=True)
    return candidates


def pick_cross_doc_pair(candidate: BridgeCandidate, chunks: dict[str, Chunk]) -> tuple[Chunk, Chunk] | None:
    """Pick two chunks from the candidate that come from different documents."""
    by_doc: dict[str, list[str]] = defaultdict(list)
    for cid in candidate.chunk_ids:
        by_doc[chunks[cid].doc_id].append(cid)
    docs = list(by_doc.keys())
    if len(docs) < 2:
        return None
    chunk_a = chunks[by_doc[docs[0]][0]]
    chunk_b = chunks[by_doc[docs[1]][0]]
    return chunk_a, chunk_b
