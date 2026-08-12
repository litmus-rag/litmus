"""Per-question-type generation logic (9 types).

Each generator takes the raw ingredients (chunk(s), an LLMClient, and any
type-specific context) and returns a plain dict with at least: "question",
"gold_answer", "intent_points" (list of {id, text, required, ...}). The
orchestrator wraps this into an ``EvalRecord`` (assigning id, gold_chunk_ids,
source_doc_ids, question_type, difficulty, etc).

Generators raise on malformed LLM output; the orchestrator is responsible
for catching and skipping/retrying a failed generation attempt so one bad
LLM response doesn't kill an entire generation run.
"""

from __future__ import annotations

import random

from litmus.generate.bridge import BridgeCandidate, pick_cross_doc_pair
from litmus.generate.prompts import (
    ADVERSARIAL_PROMPT,
    AMBIGUOUS_PROMPT,
    COMPARATIVE_PROMPT,
    COMPOUND_PROMPT,
    CONTRADICTION_PROMPT,
    CROSS_DOC_PROMPT,
    PROCEDURAL_PROMPT,
    SINGLE_CHUNK_PROMPT,
    UNANSWERABLE_PROMPT,
    WRONG_ASSUMPTION_SEED_PROMPT,
)
from litmus.llm.client import LLMClient
from litmus.models import Chunk


def generate_single_chunk(chunk: Chunk, client: LLMClient) -> dict:
    result = client.complete_json(
        SINGLE_CHUNK_PROMPT.format(chunk_text=chunk.text), temperature=0.7, max_tokens=800
    )
    result["gold_chunk_ids"] = [[chunk.id]]
    result["source_doc_ids"] = [chunk.doc_id]
    result["requires_synthesis"] = False
    return result


def generate_cross_doc(candidate: BridgeCandidate, chunks: dict[str, Chunk], client: LLMClient) -> dict | None:
    pair = pick_cross_doc_pair(candidate, chunks)
    if pair is None:
        return None
    chunk_a, chunk_b = pair
    result = client.complete_json(
        CROSS_DOC_PROMPT.format(
            bridge_entity=candidate.entity,
            doc_id_1=chunk_a.doc_id,
            chunk_text_1=chunk_a.text,
            doc_id_2=chunk_b.doc_id,
            chunk_text_2=chunk_b.text,
        ),
        temperature=0.7,
        max_tokens=900,
    )
    result["gold_chunk_ids"] = [[chunk_a.id, chunk_b.id]]
    result["source_doc_ids"] = [chunk_a.doc_id, chunk_b.doc_id]
    result["requires_synthesis"] = True
    return result


def generate_unanswerable(covered_chunks: list[Chunk], client: LLMClient) -> dict:
    sample_text = "\n".join(f"- {c.doc_id}: {c.text[:150]}" for c in covered_chunks[:8])
    result = client.complete_json(
        UNANSWERABLE_PROMPT.format(topic_sample=sample_text), temperature=0.8, max_tokens=500
    )
    result["gold_chunk_ids"] = []
    result["source_doc_ids"] = []
    result["requires_synthesis"] = False
    result["unanswerable"] = True
    return result


def generate_adversarial(
    correct_chunk: Chunk, distractor_chunks: list[Chunk], client: LLMClient
) -> dict:
    distractor_text = "\n\n".join(f"<distractor source=\"{c.doc_id}\">\n{c.text}\n</distractor>" for c in distractor_chunks)
    result = client.complete_json(
        ADVERSARIAL_PROMPT.format(
            doc_id=correct_chunk.doc_id,
            chunk_text=correct_chunk.text,
            distractor_text=distractor_text,
        ),
        temperature=0.7,
        max_tokens=800,
    )
    result["gold_chunk_ids"] = [[correct_chunk.id]]
    result["source_doc_ids"] = [correct_chunk.doc_id]
    result["requires_synthesis"] = False
    result.setdefault("scoring_notes", result.pop("must_not_include", ""))
    return result


def generate_contradiction(chunk_a: Chunk, chunk_b: Chunk, client: LLMClient) -> dict:
    result = client.complete_json(
        CONTRADICTION_PROMPT.format(
            doc_id_1=chunk_a.doc_id,
            chunk_text_1=chunk_a.text,
            doc_id_2=chunk_b.doc_id,
            chunk_text_2=chunk_b.text,
        ),
        temperature=0.7,
        max_tokens=900,
    )
    result["gold_chunk_ids"] = [[chunk_a.id, chunk_b.id]]
    result["source_doc_ids"] = [chunk_a.doc_id, chunk_b.doc_id]
    result["requires_synthesis"] = True
    return result


def generate_comparative(chunk_a: Chunk, chunk_b: Chunk, client: LLMClient) -> dict:
    result = client.complete_json(
        COMPARATIVE_PROMPT.format(
            doc_id_1=chunk_a.doc_id,
            chunk_text_1=chunk_a.text,
            doc_id_2=chunk_b.doc_id,
            chunk_text_2=chunk_b.text,
        ),
        temperature=0.7,
        max_tokens=900,
    )
    result["gold_chunk_ids"] = [[chunk_a.id, chunk_b.id]]
    result["source_doc_ids"] = [chunk_a.doc_id, chunk_b.doc_id]
    result["requires_synthesis"] = True
    return result


def generate_compound(sub_records: list[dict], client: LLMClient) -> dict:
    sub_qa_block = "\n".join(
        f"Q{i+1}: {r['question']}\nA{i+1}: {r['gold_answer']}" for i, r in enumerate(sub_records)
    )
    result = client.complete_json(
        COMPOUND_PROMPT.format(n=len(sub_records), sub_qa_block=sub_qa_block),
        temperature=0.7,
        max_tokens=900,
    )
    gold_chunk_ids = []
    source_doc_ids: list[str] = []
    for r in sub_records:
        for group in r.get("gold_chunk_ids", []):
            gold_chunk_ids.append(group)
        source_doc_ids.extend(r.get("source_doc_ids", []))
    result["gold_chunk_ids"] = gold_chunk_ids or [[]]
    result["source_doc_ids"] = list(dict.fromkeys(source_doc_ids))
    result["requires_synthesis"] = len(sub_records) > 1
    return result


def generate_procedural(chunk: Chunk, client: LLMClient) -> dict:
    result = client.complete_json(
        PROCEDURAL_PROMPT.format(chunk_text=chunk.text), temperature=0.6, max_tokens=800
    )
    result["gold_chunk_ids"] = [[chunk.id]]
    result["source_doc_ids"] = [chunk.doc_id]
    result["requires_synthesis"] = False
    return result


def generate_wrong_assumption(chunk: Chunk, client: LLMClient) -> dict:
    result = client.complete_json(
        WRONG_ASSUMPTION_SEED_PROMPT.format(chunk_text=chunk.text), temperature=0.7, max_tokens=700
    )
    result["gold_chunk_ids"] = [[chunk.id]]
    result["source_doc_ids"] = [chunk.doc_id]
    result["requires_synthesis"] = False
    result.setdefault("scoring_notes", result.pop("wrong_assumption", ""))
    return result


def generate_ambiguous(chunks: list[Chunk], client: LLMClient) -> dict:
    chunk_text = "\n\n".join(f"<passage source=\"{c.doc_id}\">\n{c.text}\n</passage>" for c in chunks)
    result = client.complete_json(
        AMBIGUOUS_PROMPT.format(chunk_text=chunk_text), temperature=0.8, max_tokens=900
    )
    result["gold_chunk_ids"] = [[c.id for c in chunks]]
    result["source_doc_ids"] = list(dict.fromkeys(c.doc_id for c in chunks))
    result["requires_synthesis"] = len(chunks) > 1
    result.setdefault("scoring_notes", str(result.pop("interpretations", "")))
    return result


def pick_distractors(
    correct_chunk: Chunk, all_chunks: dict[str, Chunk], k: int = 2, seed: int | None = None
) -> list[Chunk]:
    """Pick topically-plausible distractor chunks (same doc-type neighbors, different doc)."""
    rng = random.Random(seed)
    candidates = [c for c in all_chunks.values() if c.doc_id != correct_chunk.doc_id]
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[:k]
