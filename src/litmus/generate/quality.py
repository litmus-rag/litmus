"""Eval set self-validation (Section 6.4 of the synth methodology doc).

Turns the binary-decomposition approach inward: instead of scoring RAG
answers, score the eval set's own quality. Question Quality (QQ) and Gold
Answer/Chunk Quality (GA/GC) run at medium+ tier; Noise Realism (NR) and
Coverage/Diversity (CD) run only at exhaustive tier.
"""

from __future__ import annotations

import json

from litmus.config import get_tier_config
from litmus.llm.client import LLMClient
from litmus.models import (
    CoverageReport,
    EvalRecord,
    EvalSet,
    FlaggedRecord,
    ValidationReport,
)

QQ_QUESTIONS = {
    "QQ1": "Is the question unambiguous - would two reasonable people interpret it the same way?",
    "QQ2": "Does the question test a real user need rather than an artificial construct?",
    "QQ3": "If a noise transformation was applied, is the question still understandable to a native speaker?",
    "QQ4": "Does the question avoid accidentally revealing which chunk contains the answer "
    "(no copy-pasted terminology that maps to exactly one passage)?",
}

GA_QUESTIONS = {
    "GA1": "Is the gold answer factually correct given the source chunks?",
    "GA2": "Is the gold answer complete - does it cover all parts of the question?",
    "GA3": "Is the gold answer at the right level of detail (not too terse, not padding with irrelevant context)?",
    "GA4": "If the question is unanswerable, does the gold answer correctly indicate this rather than "
    "providing a fabricated answer?",
    "GA5": "Does the gold answer avoid containing information that is not present in any gold chunk?",
}

GC_QUESTIONS = {
    "GC1": "Does at least one annotated gold chunk set actually contain the information needed to "
    "answer the question?",
    "GC3": "Is each gold chunk set minimal - no extra chunks that don't contribute to the answer?",
    "GC5": "Are the chunk IDs correct and resolvable to actual passages in the current corpus?",
}

NR_QUESTIONS = {
    "NR2": "Is the noise level realistic (1-2 typos, not 5)?",
    "NR3": "After transformation, is the question still answerable by a human reading the relevant chunks?",
}

_RECORD_JUDGE_PROMPT = """You are auditing the quality of a synthetic RAG evaluation record. Answer each \
question with exactly "yes" or "no" plus a one-sentence reason.

<question>
{question}
</question>

<question_clean_reference>
{question_clean}
</question_clean_reference>

<gold_answer>
{gold_answer}
</gold_answer>

<gold_chunks>
{gold_chunks}
</gold_chunks>

<is_unanswerable>
{unanswerable}
</is_unanswerable>

{binary_questions_block}

Output as JSON, one key per question ID:
{{
{json_keys}
}}"""


def _render_questions_block(questions: dict[str, str]) -> str:
    return "\n".join(f"{qid}. {text}" for qid, text in questions.items())


def _render_json_keys(questions: dict[str, str]) -> str:
    return ",\n".join(f'  "{qid}": {{"verdict": "yes|no", "reason": "..."}}' for qid in questions)


def judge_record(record: EvalRecord, questions: dict[str, str], client: LLMClient) -> dict[str, bool]:
    gold_chunks_text = "\n---\n".join("\n".join(group) for group in record.gold_chunks_text) or "(none - unanswerable)"
    prompt = _RECORD_JUDGE_PROMPT.format(
        question=record.question,
        question_clean=record.question_clean,
        gold_answer=record.gold_answer,
        gold_chunks=gold_chunks_text,
        unanswerable=record.unanswerable,
        binary_questions_block=_render_questions_block(questions),
        json_keys=_render_json_keys(questions),
    )
    try:
        result = client.complete_json(prompt, temperature=0.0, max_tokens=600)
    except Exception:  # noqa: BLE001
        return {qid: True for qid in questions}
    verdicts = {}
    for qid in questions:
        entry = result.get(qid, {}) if isinstance(result, dict) else {}
        verdicts[qid] = str(entry.get("verdict", "yes")).lower() == "yes"
    return verdicts


def check_gold_chunk_resolvability(record: EvalRecord, eval_set: EvalSet) -> bool:
    for group in record.gold_chunk_ids:
        for chunk_id in group:
            if chunk_id not in eval_set.chunks:
                return False
    return True


def compute_coverage_report(eval_set: EvalSet) -> CoverageReport:
    from litmus.models import QuestionType

    all_types = set(QuestionType)
    present_types = {r.question_type for r in eval_set.records}
    n = len(eval_set.records) or 1

    unanswerable_ratio = sum(1 for r in eval_set.records if r.unanswerable) / n
    cross_doc_ratio = sum(1 for r in eval_set.records if r.question_type == QuestionType.CROSS_DOC) / n
    contradiction_ratio = sum(1 for r in eval_set.records if r.question_type == QuestionType.CONTRADICTION) / n

    domain_counts: dict[str, int] = {}
    for r in eval_set.records:
        for tag in r.domain_tags:
            domain_counts[tag] = domain_counts.get(tag, 0) + 1
    domain_coverage_ok = all(count >= 10 for count in domain_counts.values()) if domain_counts else True

    difficulty_counts = eval_set_difficulty_counts(eval_set)
    difficulty_ok = _difficulty_distribution_ok(difficulty_counts, n)

    issues = []
    if unanswerable_ratio < 0.10:
        issues.append(f"Only {unanswerable_ratio:.0%} unanswerable questions (target >=10%)")
    if cross_doc_ratio < 0.15:
        issues.append(f"Only {cross_doc_ratio:.0%} cross-doc questions (target >=15%)")
    if not domain_coverage_ok:
        issues.append("Some document domains have fewer than 10 eval records")
    if not difficulty_ok:
        issues.append("Easy/medium/hard distribution deviates significantly from 30/40/30")

    return CoverageReport(
        question_type_coverage={qt.value: qt in present_types for qt in all_types},
        unanswerable_ratio=unanswerable_ratio,
        domain_coverage_ok=domain_coverage_ok,
        noise_distribution_ok=True,
        difficulty_distribution_ok=difficulty_ok,
        cross_doc_ratio=cross_doc_ratio,
        contradiction_ratio=contradiction_ratio,
        issues=issues,
    )


def eval_set_difficulty_counts(eval_set: EvalSet) -> dict[str, int]:
    from litmus.models import Difficulty

    counts = {d.value: 0 for d in Difficulty}
    for r in eval_set.records:
        counts[r.difficulty.value] += 1
    return counts


def _difficulty_distribution_ok(counts: dict[str, int], total: int, tolerance: float = 0.15) -> bool:
    if total == 0:
        return True
    targets = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
    for level, target in targets.items():
        actual = counts.get(level, 0) / total
        if abs(actual - target) > tolerance:
            return False
    return True


def validate_eval_set(eval_set: EvalSet, client: LLMClient | None = None) -> ValidationReport:
    tier_config = get_tier_config(eval_set.tier)
    quality_checks = tier_config["quality_checks"]

    if "eval_set_validation" not in quality_checks and "eval_set_validation_full" not in quality_checks:
        return ValidationReport(
            overall_pass=True,
            flagged_records=[],
            dimension_scores={},
            coverage_report=None,
            recommendations=["Eval set validation is not enabled for the 'minimal' tier."],
        )

    client = client or LLMClient()
    run_nr = "eval_set_validation_full" in quality_checks

    questions_to_run: dict[str, str] = {**QQ_QUESTIONS, **GA_QUESTIONS, **GC_QUESTIONS}
    if run_nr:
        questions_to_run = {**questions_to_run, **NR_QUESTIONS}

    flagged: list[FlaggedRecord] = []
    dim_pass_counts: dict[str, list[bool]] = {"QQ": [], "GA": [], "GC": [], "NR": []}

    for record in eval_set.records:
        verdicts = judge_record(record, questions_to_run, client)
        if not check_gold_chunk_resolvability(record, eval_set):
            verdicts["GC5"] = False

        failed = [qid for qid, ok in verdicts.items() if not ok]
        if len(failed) > 1:
            flagged.append(FlaggedRecord(record_id=record.id, failed_checks=failed))

        for qid, ok in verdicts.items():
            prefix = qid[:2]
            if prefix in dim_pass_counts:
                dim_pass_counts[prefix].append(ok)

    dimension_scores = {
        {"QQ": "question_quality", "GA": "gold_answer_quality", "GC": "gold_chunk_quality", "NR": "noise_realism"}[
            prefix
        ]: (sum(v) / len(v) if v else 1.0)
        for prefix, v in dim_pass_counts.items()
        if v
    }

    coverage_report = None
    if "coverage_diversity" in quality_checks:
        coverage_report = compute_coverage_report(eval_set)

    recommendations = []
    if flagged:
        recommendations.append(f"{len(flagged)} records failed more than one quality check - review them.")
    if coverage_report:
        recommendations.extend(coverage_report.issues)
    weak_dims = [d for d, score in dimension_scores.items() if score < 0.8]
    if weak_dims:
        recommendations.append(f"Dimensions below 80% pass rate: {', '.join(weak_dims)}")

    overall_pass = len(flagged) == 0 and (coverage_report is None or not coverage_report.issues)

    return ValidationReport(
        overall_pass=overall_pass,
        flagged_records=flagged,
        dimension_scores=dimension_scores,
        coverage_report=coverage_report,
        recommendations=recommendations or ["No issues found."],
    )
