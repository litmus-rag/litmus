"""Binary decomposition answer scoring (Section 6.2 of the synth doc).

Builds the tier's default 9- or 24-question set (or accepts a fully custom
``list[BinaryQuestion]``), runs one judge LLM call per record, and
aggregates verdicts into per-dimension ``DimensionScore`` plus a weighted
``overall_score``.
"""

from __future__ import annotations

from litmus.config import get_default_weights
from litmus.evaluate.prompts import build_judge_prompt
from litmus.llm.client import LLMClient
from litmus.models import BinaryQuestion, BinaryVerdict, DimensionScore

MVP_QUESTIONS: list[BinaryQuestion] = [
    BinaryQuestion("F1", "faithfulness", "Is every factual claim in the generated answer supported by content in the retrieved context?"),
    BinaryQuestion("F2", "faithfulness", "Is the generated answer free of fabricated entities, numbers, or facts not present in the retrieved context?"),
    BinaryQuestion("F3", "faithfulness", "Does the generated answer avoid misattributing statements or facts to the wrong entity, source, or time period?"),
    BinaryQuestion("C1", "correctness", "Does the generated answer address the specific question that was asked (not a related but different question)?"),
    BinaryQuestion("C2", "correctness", "Are the core facts in the generated answer consistent with the gold answer?"),
    BinaryQuestion("C3", "correctness", "If the question has multiple parts, does the generated answer address all of them? (If single-part, answer yes.)"),
    BinaryQuestion("A1", "abstention", "If the question is unanswerable, does the generated answer clearly indicate it cannot be answered from the available documentation? (If answerable, answer yes.)"),
    BinaryQuestion("A2", "abstention", "If the question is answerable, does the generated answer provide a substantive response rather than refusing unnecessarily? (If unanswerable, answer yes.)"),
    BinaryQuestion("A3", "abstention", "Does the generated answer express appropriate uncertainty when the retrieved evidence is ambiguous or conflicting? (If unambiguous, answer yes.)"),
]

EXHAUSTIVE_QUESTIONS: list[BinaryQuestion] = [
    BinaryQuestion("F1", "faithfulness", "Is every factual claim in the generated answer supported by content in the retrieved context?"),
    BinaryQuestion("F2", "faithfulness", "Is the generated answer free of fabricated information not present in any retrieved chunk?"),
    BinaryQuestion("F3", "faithfulness", "Are all named entities (people, orgs, locations) accurately represented as in the source chunks?"),
    BinaryQuestion("F4", "faithfulness", "Are all numerical claims (dates, quantities, statistics) consistent with the source chunks?"),
    BinaryQuestion("F5", "faithfulness", "Are causal relationships and event sequences preserved accurately from the source?"),
    BinaryQuestion("F6", "faithfulness", "Does the answer avoid misattributing statements or actions to the wrong entity?"),
    BinaryQuestion("F7", "faithfulness", "Does the answer avoid conflating information from separate chunks in a way that distorts meaning?"),
    BinaryQuestion("F8", "faithfulness", "When the answer cites or attributes a claim to a source, is that source actually the one containing the claim?"),
    BinaryQuestion("C1", "correctness", "Does the answer address the question that was actually asked?"),
    BinaryQuestion("C2", "correctness", "Are the core facts in the answer consistent with the gold answer?"),
    BinaryQuestion("C3", "correctness", "Does the answer avoid stating the opposite of what the gold answer says?"),
    BinaryQuestion("C4", "correctness", "If multi-part, does the answer address all parts? (If single-part, answer yes.)"),
    BinaryQuestion("C5", "correctness", "If comparative, does the answer preserve the correct direction of comparison? (If not comparative, answer yes.)"),
    BinaryQuestion("CP1", "completeness", "Does the answer cover the most important point from the gold answer?"),
    BinaryQuestion("CP2", "completeness", "Does the answer include key supporting details, not just the headline fact?"),
    BinaryQuestion("CP3", "completeness", "If the gold answer mentions qualifications/caveats, does the answer include them? (If none, answer yes.)"),
    BinaryQuestion("CP4", "completeness", "If cross-document synthesis is required, does the answer integrate information from all required sources? (If not required, answer yes.)"),
    BinaryQuestion("CP5", "completeness", "If conflicting sources are involved, does the answer surface the conflict rather than silently picking one? (If not applicable, answer yes.)"),
    BinaryQuestion("CN1", "conciseness", "Is the answer free of information that doesn't help answer the question?"),
    BinaryQuestion("CN2", "conciseness", "Does the answer avoid repeating the same point in different words?"),
    BinaryQuestion("CN3", "conciseness", "Does the answer avoid dumping raw chunk content without synthesizing it?"),
    BinaryQuestion("A1", "abstention", "If unanswerable, does the answer clearly say so? (If answerable, answer yes.)"),
    BinaryQuestion("A2", "abstention", "If answerable, does the answer provide a substantive response? (If unanswerable, answer yes.)"),
    BinaryQuestion("A3", "abstention", "Does the answer express appropriate uncertainty when evidence is ambiguous? (If unambiguous, answer yes.)"),
]


def default_questions_for_tier(tier: str, scoring: str | None = None) -> list[BinaryQuestion]:
    if scoring == "mvp":
        return MVP_QUESTIONS
    if scoring == "full":
        return EXHAUSTIVE_QUESTIONS
    return EXHAUSTIVE_QUESTIONS if tier == "exhaustive" else MVP_QUESTIONS


def resolve_questions(
    tier: str,
    scoring: str | list[BinaryQuestion] | None,
) -> list[BinaryQuestion]:
    if isinstance(scoring, list):
        if not scoring:
            raise ValueError("Custom scoring question list cannot be empty")
        return scoring
    return default_questions_for_tier(tier, scoring)


def resolve_weights(
    tier: str,
    questions: list[BinaryQuestion],
    weights: dict[str, float] | None,
    is_custom: bool,
) -> dict[str, float]:
    dimensions = sorted({q.dimension for q in questions})
    if weights:
        missing = set(dimensions) - set(weights)
        if missing:
            raise ValueError(f"weights is missing entries for dimensions: {missing}")
        return weights
    if is_custom:
        # No explicit weights given for a custom question set: weight dimensions equally.
        return {d: 1.0 / len(dimensions) for d in dimensions}
    return get_default_weights(tier)


def run_judge(
    question: str,
    retrieved_context: list[str],
    generated_answer: str,
    gold_answer: str,
    is_unanswerable: bool,
    questions: list[BinaryQuestion],
    client: LLMClient,
) -> dict[str, BinaryVerdict]:
    prompt = build_judge_prompt(question, retrieved_context, generated_answer, gold_answer, is_unanswerable, questions)
    try:
        raw = client.complete_json(prompt, temperature=0.0, max_tokens=1200)
    except Exception:  # noqa: BLE001
        raw = {}
    verdicts: dict[str, BinaryVerdict] = {}
    for q in questions:
        entry = raw.get(q.id, {}) if isinstance(raw, dict) else {}
        verdict_bool = str(entry.get("verdict", "no")).strip().lower() == "yes"
        reason = str(entry.get("reason", ""))
        verdicts[q.id] = BinaryVerdict(question_id=q.id, verdict=verdict_bool, reason=reason)
    return verdicts


def aggregate_dimension_scores(
    verdicts: dict[str, BinaryVerdict], questions: list[BinaryQuestion]
) -> dict[str, DimensionScore]:
    by_dimension: dict[str, list[BinaryQuestion]] = {}
    for q in questions:
        by_dimension.setdefault(q.dimension, []).append(q)

    scores: dict[str, DimensionScore] = {}
    for dimension, qs in by_dimension.items():
        total_weight = sum(q.weight for q in qs) or 1.0
        weighted_yes = sum(q.weight for q in qs if verdicts.get(q.id, BinaryVerdict(q.id, False)).verdict)
        dim_verdicts = {q.id: verdicts[q.id] for q in qs if q.id in verdicts}
        scores[dimension] = DimensionScore(score=weighted_yes / total_weight, verdicts=dim_verdicts)
    return scores


def compute_overall_score(dimension_scores: dict[str, DimensionScore], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1.0
    weighted_sum = sum(dimension_scores[d].score * w for d, w in weights.items() if d in dimension_scores)
    return weighted_sum / total_weight
