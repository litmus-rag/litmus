"""Full evaluate() pipeline: run RAG, score retrieval, score answers, score
intent coverage, discover alternative chunks, checkpoint per record.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from litmus.cache.manager import CacheManager
from litmus.config import get_tier_config
from litmus.evaluate.answer_scorer import (
    aggregate_dimension_scores,
    compute_overall_score,
    resolve_questions,
    resolve_weights,
    run_judge,
)
from litmus.evaluate.chunk_discovery import apply_discovered_alternatives, discover_alternative_chunks
from litmus.evaluate.diagnostics import detect_stub_rag
from litmus.evaluate.intent_scorer import score_intent_coverage
from litmus.evaluate.retrieval_scorer import score_retrieval
from litmus.evaluate.runner import run_rag_sync
from litmus.llm.client import LLMClient
from litmus.models import BinaryQuestion, EvalSet, EvalResults, RecordResult


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[litmus.evaluate] {message}")


def _checkpoint_key(eval_set: EvalSet, scoring) -> str:
    """Fingerprint identifying this specific eval set + scoring config, so
    checkpoint keys never collide across two different evaluate() calls
    sharing a cache_dir."""
    fingerprint = f"{eval_set.metadata.docs_dir}|{eval_set.metadata.generated_at}|{eval_set.tier}|{scoring}"
    return "evaluate:results:" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def _evaluate_one_record(
    record,
    rag: Callable,
    judge_client: LLMClient,
    intent_client: LLMClient,
    questions: list[BinaryQuestion],
    weights: dict[str, float],
    retrieval_metrics: list[str],
    timeout: int,
) -> RecordResult:
    rag_response, rag_error = run_rag_sync(rag, record.question, timeout=timeout)

    retrieval = score_retrieval(
        record.gold_chunks_text,
        rag_response.contexts,
        compute_mrr="mrr" in retrieval_metrics,
        compute_precision="precision_at_k" in retrieval_metrics,
        compute_ranks="gold_chunk_ranks" in retrieval_metrics,
    )

    verdicts = run_judge(
        record.question,
        rag_response.contexts,
        rag_response.answer,
        record.gold_answer,
        record.unanswerable,
        questions,
        judge_client,
    )
    dimension_scores = aggregate_dimension_scores(verdicts, questions)
    overall = compute_overall_score(dimension_scores, weights)

    intent_coverage, intent_details = score_intent_coverage(record.intent_points, rag_response.answer, intent_client)

    discovered = discover_alternative_chunks(record, rag_response.contexts, overall, retrieval.set_recall)
    if discovered:
        apply_discovered_alternatives(record, discovered)

    return RecordResult(
        record_id=record.id,
        question=record.question,
        question_type=record.question_type,
        noise_profile=record.noise_profile,
        difficulty=record.difficulty,
        rag_answer=rag_response.answer,
        rag_contexts=rag_response.contexts,
        rag_error=rag_error,
        chunk_recall=retrieval.chunk_recall,
        set_recall=retrieval.set_recall,
        mrr=retrieval.mrr,
        precision_at_k=retrieval.precision_at_k,
        gold_chunk_ranks=retrieval.gold_chunk_ranks,
        faithfulness=dimension_scores.get("faithfulness"),
        correctness=dimension_scores.get("correctness"),
        abstention=dimension_scores.get("abstention"),
        completeness=dimension_scores.get("completeness"),
        conciseness=dimension_scores.get("conciseness"),
        overall_score=overall,
        intent_coverage=intent_coverage,
        intent_details=intent_details,
        discovered_alt_chunks=discovered,
        domain_tags=record.domain_tags,
    )


def evaluate_eval_set(
    eval_set: EvalSet,
    rag: Callable,
    llm: str | None = None,
    scoring: str | list[BinaryQuestion] | None = None,
    weights: dict[str, float] | None = None,
    save_path: str | None = None,
    max_workers: int = 1,
    timeout: int = 60,
    cache_dir: str | None = ".litmus_cache",
    verbose: bool = True,
) -> EvalResults:
    tier_config = get_tier_config(eval_set.tier)
    model = llm or eval_set.metadata.llm or "azure/gpt-5.4"
    judge_client = LLMClient(model=model)
    intent_client = LLMClient(model=model)
    cache = CacheManager(cache_dir)

    if cache_dir:
        _log(verbose, f"Cache/checkpoint directory: {Path(cache_dir).resolve()}")

    is_custom = isinstance(scoring, list)
    questions = resolve_questions(eval_set.tier, scoring)
    resolved_weights = resolve_weights(eval_set.tier, questions, weights, is_custom)
    retrieval_metrics = tier_config["retrieval_metrics"]

    _log(verbose, f"Evaluating {len(eval_set.records)} records (tier={eval_set.tier}, model={model})")

    # Scope the checkpoint to this specific eval set so two different
    # evaluate() calls sharing a cache_dir never resume each other's scored
    # records -- a flat "evaluate:results" key previously collided across
    # any two eval sets evaluated into the same cache_dir, silently
    # splicing one eval set's scored records into another's results.
    checkpoint_key = _checkpoint_key(eval_set, scoring)
    cached_results = cache.get_checkpoint(checkpoint_key, [])
    done_ids = {r["record_id"] for r in cached_results}
    pending = [r for r in eval_set.records if r.id not in done_ids]

    if done_ids:
        _log(verbose, f"Resuming: {len(done_ids)} already scored, {len(pending)} remaining")

    def _score(record):
        return _evaluate_one_record(
            record, rag, judge_client, intent_client, questions, resolved_weights, retrieval_metrics, timeout
        )

    new_results: list[RecordResult] = []
    if max_workers <= 1 or len(pending) <= 1:
        for i, record in enumerate(pending):
            new_results.append(_score(record))
            if (i + 1) % 10 == 0:
                _log(verbose, f"  scored {i + 1}/{len(pending)}")
                cache.set_checkpoint(checkpoint_key, cached_results + [r.to_dict() for r in new_results])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score, record): record for record in pending}
            completed = 0
            for future in as_completed(futures):
                new_results.append(future.result())
                completed += 1
                if completed % 10 == 0:
                    _log(verbose, f"  scored {completed}/{len(pending)}")

    all_results = [RecordResult.from_dict(r) for r in cached_results] + new_results
    cache.clear_checkpoint(checkpoint_key)

    stub_warnings = detect_stub_rag(all_results)
    for warning in stub_warnings:
        _log(True, f"WARNING: {warning}")

    results = EvalResults(
        records=all_results,
        tier=eval_set.tier,
        metadata={
            "llm": model,
            "scoring": "custom" if is_custom else (scoring or "default"),
            "weights": resolved_weights,
            "stub_rag_warnings": stub_warnings,
        },
    )

    if save_path:
        results.to_json(save_path)
        _log(verbose, f"Saved results to {Path(save_path).resolve()}")
    else:
        _log(verbose, "No save_path given - results were NOT written to disk (only returned in memory).")

    _log(verbose, f"Done: {len(all_results)} records evaluated.")
    return results
