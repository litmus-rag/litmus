"""Leakage filter (SeedRG-style knowledge-leakage check).

Runs each generated question through the LLM with no retrieved context. If
the LLM answers correctly from parametric memory alone, the question is
discarded — it doesn't test retrieval, it tests what the model already
knows. Most useful for corpora built from public content; internal/private
corpora (like the pharma drug labels in sample1/) will rarely trigger this,
but the check costs one cheap LLM call per question and catches surprises
(e.g. a well-known public drug label the model has memorized).
"""

from __future__ import annotations

from litmus.generate.prompts import LEAKAGE_CHECK_PROMPT
from litmus.llm.client import LLMClient
from litmus.models import EvalRecord


def check_leakage(question: str, gold_answer: str, client: LLMClient) -> bool:
    """Return True if the question appears answerable from parametric memory alone."""
    if not gold_answer.strip():
        return False
    prompt = LEAKAGE_CHECK_PROMPT.format(question=question)
    try:
        result = client.complete_json(prompt, temperature=0.0, max_tokens=300)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(result, dict):
        return False
    return bool(result.get("can_answer_from_memory", False))


def filter_leaked_records(records: list[EvalRecord], client: LLMClient) -> tuple[list[EvalRecord], list[str]]:
    """Split records into (kept, discarded_ids) based on the leakage check.

    Unanswerable records are exempt (there's no gold answer to leak).
    """
    kept: list[EvalRecord] = []
    discarded: list[str] = []
    for record in records:
        if record.unanswerable:
            kept.append(record)
            continue
        if check_leakage(record.question_clean, record.gold_answer, client):
            discarded.append(record.id)
        else:
            kept.append(record)
    return kept, discarded
