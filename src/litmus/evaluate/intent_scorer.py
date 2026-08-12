"""Intent point coverage scoring (Section B.13 of the synth doc).

For each intent point, an LLM judge asks "does the generated answer
clearly convey this point?" Aggregated with required points weighted 1.0
and preferred points weighted 0.5, per the formula in the spec:

    intent_coverage = (required_covered + 0.5 * preferred_covered) / total_weighted_points
"""

from __future__ import annotations

from litmus.evaluate.prompts import INTENT_COVERAGE_PROMPT
from litmus.llm.client import LLMClient
from litmus.models import IntentPoint, IntentVerdict


def check_intent_point(intent_point: IntentPoint, generated_answer: str, client: LLMClient) -> IntentVerdict:
    prompt = INTENT_COVERAGE_PROMPT.format(intent_point_text=intent_point.text, generated_answer=generated_answer)
    try:
        result = client.complete_json(prompt, temperature=0.0, max_tokens=150)
    except Exception:  # noqa: BLE001
        return IntentVerdict(intent_point_id=intent_point.id, covered=False, reason="judge call failed")
    covered = bool(result.get("covered", False)) if isinstance(result, dict) else False
    reason = str(result.get("reason", "")) if isinstance(result, dict) else ""
    return IntentVerdict(intent_point_id=intent_point.id, covered=covered, reason=reason)


def score_intent_coverage(
    intent_points: list[IntentPoint], generated_answer: str, client: LLMClient
) -> tuple[float, list[IntentVerdict]]:
    if not intent_points:
        return 1.0, []
    verdicts = [check_intent_point(p, generated_answer, client) for p in intent_points]
    verdict_by_id = {v.intent_point_id: v for v in verdicts}

    total_weight = 0.0
    covered_weight = 0.0
    for point in intent_points:
        weight = 1.0 if point.required else 0.5
        total_weight += weight
        verdict = verdict_by_id.get(point.id)
        if verdict and verdict.covered:
            covered_weight += weight

    coverage = covered_weight / total_weight if total_weight else 1.0
    return coverage, verdicts
