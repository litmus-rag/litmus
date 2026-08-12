"""Contradiction detection (exhaustive tier ingestion step).

Pipeline: embed all chunks, pre-filter to candidate pairs above a cosine
similarity threshold (cheap, no LLM), restricted to cross-document pairs
(same-doc "contradictions" are usually just elaboration, not conflict).
Each surviving candidate pair is then confirmed with one LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from litmus.config import CONTRADICTION_SIMILARITY_THRESHOLD
from litmus.ingest.embedder import embed_texts, top_similar_pairs
from litmus.llm.client import LLMClient
from litmus.models import Chunk

_CONTRADICTION_CONFIRM_PROMPT = """Do these two passages give different answers to the same underlying \
question, such that a user asking that question would get a different answer depending on which passage \
they read? This includes:
- Different numbers, thresholds, or policies for the same thing
- Mutually exclusive statements
- One passage stating an older version of a policy that a newer passage supersedes or amends (this counts \
as a contradiction even if one document explicitly says it supersedes the other - a naive retrieval system \
could still surface the outdated passage and mislead a user)

Passages that simply cover different, unrelated topics are NOT contradictions.

<passage_1>
{text_1}
</passage_1>

<passage_2>
{text_2}
</passage_2>

Output as JSON:
{{"contradiction": true or false, "reason": "one sentence explanation"}}"""


@dataclass
class ContradictionPair:
    chunk_id_a: str
    chunk_id_b: str
    similarity: float
    reason: str = ""


def confirm_contradiction(chunk_a: Chunk, chunk_b: Chunk, client: LLMClient) -> tuple[bool, str]:
    prompt = _CONTRADICTION_CONFIRM_PROMPT.format(text_1=chunk_a.text, text_2=chunk_b.text)
    try:
        result = client.complete_json(prompt, temperature=0.0, max_tokens=200)
    except Exception:  # noqa: BLE001
        return False, ""
    if not isinstance(result, dict):
        return False, ""
    return bool(result.get("contradiction", False)), str(result.get("reason", ""))


def find_contradictions(
    chunks: dict[str, Chunk],
    client: LLMClient,
    similarity_threshold: float = CONTRADICTION_SIMILARITY_THRESHOLD,
    max_candidates: int = 50,
) -> list[ContradictionPair]:
    """Find contradiction pairs across the corpus.

    Returns an empty list (with the caller expected to log a warning) if no
    natural contradictions are found - some corpora are simply consistent.
    """
    ids = list(chunks.keys())
    if len(ids) < 2:
        return []
    texts = [chunks[cid].text for cid in ids]
    embeddings = embed_texts(texts)
    doc_map = {cid: chunks[cid].doc_id for cid in ids}
    candidates = top_similar_pairs(embeddings, ids, threshold=similarity_threshold, exclude_same_doc=doc_map)
    candidates = candidates[:max_candidates]

    confirmed: list[ContradictionPair] = []
    for id_a, id_b, sim in candidates:
        is_contradiction, reason = confirm_contradiction(chunks[id_a], chunks[id_b], client)
        if is_contradiction:
            confirmed.append(ContradictionPair(chunk_id_a=id_a, chunk_id_b=id_b, similarity=sim, reason=reason))
    return confirmed
