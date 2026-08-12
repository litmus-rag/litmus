"""Scoring health diagnostics (Section 6.3 of the synth doc): yes-rate
spread, phi-coefficient matrix, failure mode coverage. Also backs
EvalResults.failure_patterns() and .retrieval_summary().

All computation here is pure math over already-scored RecordResults - no
LLM calls, fully deterministic, easy to unit test.
"""

from __future__ import annotations

import math

from litmus.config import (
    FAILURE_MODES,
    HIGH_CORR_PHI_THRESHOLD,
    MEAN_PHI_HEALTHY,
    MEAN_PHI_NEEDS_WORK,
    WEAK_PAIR_HEALTHY_PCT,
    WEAK_PAIR_PHI_THRESHOLD,
    YES_RATE_MAX_TARGET,
    YES_RATE_MIN_TARGET,
    YES_RATE_SPREAD_HEALTHY,
    YES_RATE_SPREAD_NEEDS_WORK,
)
from litmus.models import EvalResults, FailurePatternReport, RecordResult, RetrievalReport, ScoringHealthReport

# Exact strings from the README/spec quickstart stub RAG. If a real RAG
# ever legitimately returns one of these verbatim, the warning is a
# harmless false positive - but that's vanishingly unlikely for real
# retrieval output.
_KNOWN_PLACEHOLDER_ANSWERS = {"...", "", "todo", "answer", "your answer here"}
_KNOWN_PLACEHOLDER_CONTEXTS = {"chunk text 1", "chunk text 2", "...", "context", "your context here"}


def detect_stub_rag(results: list[RecordResult]) -> list[str]:
    """Heuristically detect a RAG callable that was never actually implemented.

    Looks for two independent signals, either of which is enough to warn:
    1. The exact placeholder text from the quickstart example (`"..."`,
       `"chunk text 1"`/`"chunk text 2"`, etc).
    2. The RAG returned byte-identical (answer, contexts) for every single
       question - real retrieval varies with the question; a constant
       response means nothing upstream of the return statement ran.

    Returns a list of human-readable warnings (empty if nothing looks stubbed).
    """
    if not results:
        return []

    warnings: list[str] = []

    placeholder_hits = sum(
        1
        for r in results
        if r.rag_answer.strip().lower() in _KNOWN_PLACEHOLDER_ANSWERS
        or any(ctx.strip().lower() in _KNOWN_PLACEHOLDER_CONTEXTS for ctx in r.rag_contexts)
    )
    if placeholder_hits:
        warnings.append(
            f"{placeholder_hits}/{len(results)} record(s) returned placeholder-looking text "
            '(e.g. "...", "chunk text 1") instead of a real answer/context - '
            "your `rag` callable looks like it's still the quickstart stub, not a real RAG system."
        )

    if len(results) >= 2:
        distinct_responses = {(r.rag_answer, tuple(r.rag_contexts)) for r in results}
        if len(distinct_responses) == 1:
            warnings.append(
                f"All {len(results)} questions received the exact same answer and contexts. "
                "A real RAG system's output should vary with the question - check that `rag` "
                "is actually retrieving and generating, not returning a constant."
            )

    return warnings


def _all_dimension_scores(results: EvalResults):
    for r in results.records:
        yield r.faithfulness
        yield r.correctness
        yield r.abstention
        if r.completeness:
            yield r.completeness
        if r.conciseness:
            yield r.conciseness


def compute_yes_rates(results: EvalResults) -> dict[str, float]:
    """Per binary-question-id yes-rate across all records."""
    yes_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for dim_score in _all_dimension_scores(results):
        for qid, verdict in dim_score.verdicts.items():
            total_counts[qid] = total_counts.get(qid, 0) + 1
            if verdict.verdict:
                yes_counts[qid] = yes_counts.get(qid, 0) + 1
    return {qid: yes_counts.get(qid, 0) / total for qid, total in total_counts.items()}


def _question_dimension_map(results: EvalResults) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for r in results.records:
        for dim_name, dim_score in (
            ("faithfulness", r.faithfulness),
            ("correctness", r.correctness),
            ("abstention", r.abstention),
            ("completeness", r.completeness),
            ("conciseness", r.conciseness),
        ):
            if dim_score is None:
                continue
            for qid in dim_score.verdicts:
                mapping[qid] = dim_name
    return mapping


def compute_yes_rate_spread_per_dimension(results: EvalResults) -> dict[str, float]:
    yes_rates = compute_yes_rates(results)
    dim_map = _question_dimension_map(results)
    by_dim: dict[str, list[float]] = {}
    for qid, rate in yes_rates.items():
        dim = dim_map.get(qid)
        if dim:
            by_dim.setdefault(dim, []).append(rate)
    return {dim: (max(rates) - min(rates)) for dim, rates in by_dim.items() if rates}


def _phi_coefficient(a_verdicts: list[bool], b_verdicts: list[bool]) -> float:
    n11 = sum(1 for a, b in zip(a_verdicts, b_verdicts) if a and b)
    n00 = sum(1 for a, b in zip(a_verdicts, b_verdicts) if not a and not b)
    n10 = sum(1 for a, b in zip(a_verdicts, b_verdicts) if a and not b)
    n01 = sum(1 for a, b in zip(a_verdicts, b_verdicts) if not a and b)
    n1_ = n11 + n10
    n0_ = n01 + n00
    n_1 = n11 + n01
    n_0 = n10 + n00
    denom = n1_ * n0_ * n_1 * n_0
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / math.sqrt(denom)


def _collect_verdict_series(results: EvalResults) -> dict[str, list[bool]]:
    series: dict[str, list[bool]] = {}
    for r in results.records:
        for dim_score in (r.faithfulness, r.correctness, r.abstention, r.completeness, r.conciseness):
            if dim_score is None:
                continue
            for qid, verdict in dim_score.verdicts.items():
                series.setdefault(qid, []).append(verdict.verdict)
    return series


def compute_phi_matrix(results: EvalResults) -> dict[str, dict[str, float]]:
    series = _collect_verdict_series(results)
    dim_map = _question_dimension_map(results)
    by_dim: dict[str, list[str]] = {}
    for qid, dim in dim_map.items():
        by_dim.setdefault(dim, []).append(qid)

    matrix: dict[str, dict[str, float]] = {}
    for _dim, qids in by_dim.items():
        for qid_a in qids:
            matrix.setdefault(qid_a, {})
            for qid_b in qids:
                if qid_a == qid_b:
                    matrix[qid_a][qid_b] = 1.0
                    continue
                a_series = series.get(qid_a, [])
                b_series = series.get(qid_b, [])
                n = min(len(a_series), len(b_series))
                if n == 0:
                    matrix[qid_a][qid_b] = 0.0
                else:
                    matrix[qid_a][qid_b] = _phi_coefficient(a_series[:n], b_series[:n])
    return matrix


def high_correlation_pairs(phi_matrix: dict[str, dict[str, float]], threshold: float = HIGH_CORR_PHI_THRESHOLD) -> list[tuple[str, str, float]]:
    seen = set()
    pairs = []
    for qid_a, row in phi_matrix.items():
        for qid_b, phi in row.items():
            if qid_a == qid_b:
                continue
            key = frozenset((qid_a, qid_b))
            if key in seen:
                continue
            seen.add(key)
            if abs(phi) > threshold:
                pairs.append((qid_a, qid_b, phi))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs


def _mean_off_diagonal_phi(phi_matrix: dict[str, dict[str, float]]) -> float:
    values = [abs(phi) for qid_a, row in phi_matrix.items() for qid_b, phi in row.items() if qid_a != qid_b]
    return sum(values) / len(values) if values else 0.0


def _pct_weak_pairs(phi_matrix: dict[str, dict[str, float]], threshold: float = WEAK_PAIR_PHI_THRESHOLD) -> float:
    values = [abs(phi) for qid_a, row in phi_matrix.items() for qid_b, phi in row.items() if qid_a != qid_b]
    if not values:
        return 1.0
    return sum(1 for v in values if v < threshold) / len(values)


def detect_uncovered_failure_modes(results: EvalResults) -> list[str]:
    """Heuristic coverage check: a failure mode is "uncovered" if the yes-rate
    of every binary question that could plausibly catch it never dips below
    0.98 across the whole eval set (i.e. nothing ever fails there)."""
    yes_rates = compute_yes_rates(results)
    if not yes_rates:
        return list(FAILURE_MODES)
    min_yes_rate = min(yes_rates.values())
    # If literally nothing ever fails on any binary question, no failure
    # mode is being caught in practice, regardless of question wording.
    if min_yes_rate >= 0.98:
        return list(FAILURE_MODES)
    return []


def compute_scoring_health(results: EvalResults) -> ScoringHealthReport:
    yes_rates = compute_yes_rates(results)
    spread_per_dim = compute_yes_rate_spread_per_dimension(results)
    phi_matrix = compute_phi_matrix(results)
    high_corr = high_correlation_pairs(phi_matrix)
    uncovered = detect_uncovered_failure_modes(results)

    mean_phi = _mean_off_diagonal_phi(phi_matrix)
    pct_weak = _pct_weak_pairs(phi_matrix)
    min_spread = min(spread_per_dim.values()) if spread_per_dim else 0.0

    issues = []
    if min_spread < YES_RATE_SPREAD_NEEDS_WORK:
        verdict = "broken"
        issues.append(f"Yes-rate spread as low as {min_spread:.2f} in at least one dimension (target >0.20)")
    elif min_spread < YES_RATE_SPREAD_HEALTHY:
        verdict = "needs_work"
        issues.append(f"Yes-rate spread of {min_spread:.2f} below healthy threshold in at least one dimension")
    else:
        verdict = "healthy"

    if mean_phi > MEAN_PHI_NEEDS_WORK:
        verdict = "broken"
        issues.append(f"Mean off-diagonal phi is {mean_phi:.2f} (target <0.40)")
    elif mean_phi > MEAN_PHI_HEALTHY and verdict == "healthy":
        verdict = "needs_work"
        issues.append(f"Mean off-diagonal phi is {mean_phi:.2f} (target <0.40)")

    if pct_weak < WEAK_PAIR_HEALTHY_PCT and verdict == "healthy":
        verdict = "needs_work"
        issues.append(f"Only {pct_weak:.0%} of question pairs have |phi| < 0.3 (target >50%)")

    for qid, rate in yes_rates.items():
        if rate > YES_RATE_MAX_TARGET:
            issues.append(f"{qid} yes-rate {rate:.2f} is too easy (>0.95), not discriminating")
        elif rate < 1 - YES_RATE_MAX_TARGET and rate < YES_RATE_MIN_TARGET:
            issues.append(f"{qid} yes-rate {rate:.2f} is too strict (<0.10), everything fails")

    if uncovered:
        if verdict == "healthy":
            verdict = "needs_work"
        issues.append(f"{len(uncovered)} failure modes potentially uncovered - nothing ever fails a binary question")

    if high_corr:
        issues.append(f"{len(high_corr)} question pairs are near-duplicates (phi > 0.7): consider merging")

    return ScoringHealthReport(
        yes_rate_per_question=yes_rates,
        yes_rate_spread_per_dimension=spread_per_dim,
        phi_matrix=phi_matrix,
        high_correlation_pairs=high_corr,
        uncovered_failure_modes=uncovered,
        verdict=verdict,
        recommendations=issues or ["Scoring health looks good."],
    )


def compute_failure_patterns(results: EvalResults) -> FailurePatternReport:
    yes_rates = compute_yes_rates(results)
    worst = sorted(yes_rates.items(), key=lambda kv: kv[1])[:10]

    patterns: list[dict] = []
    by_type = results.by_question_type()
    for qtype, agg in sorted(by_type.items(), key=lambda kv: kv[1].mean_overall):
        if agg.mean_overall < 0.8:
            patterns.append(
                {
                    "question_type": qtype.value,
                    "mean_overall": agg.mean_overall,
                    "count": agg.count,
                }
            )
    return FailurePatternReport(top_patterns=patterns, worst_binary_questions=worst)


def compute_retrieval_summary(results: EvalResults) -> RetrievalReport:
    records = results.records
    n = len(records) or 1
    mean_set_recall = sum(1.0 if r.set_recall else 0.0 for r in records) / n
    mean_chunk_recall = sum(r.chunk_recall for r in records) / n
    mrr_values = [r.mrr for r in records if r.mrr is not None]
    precision_values = [r.precision_at_k for r in records if r.precision_at_k is not None]

    by_type = {}
    for qtype, agg in results.by_question_type().items():
        by_type[qtype.value] = agg.mean_set_recall or 0.0

    return RetrievalReport(
        mean_set_recall=mean_set_recall,
        mean_chunk_recall=mean_chunk_recall,
        mean_mrr=(sum(mrr_values) / len(mrr_values)) if mrr_values else None,
        mean_precision_at_k=(sum(precision_values) / len(precision_values)) if precision_values else None,
        by_question_type=by_type,
    )
