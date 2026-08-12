"""Top-level orchestration for the public API (generate, evaluate, estimate_cost, load)."""

from __future__ import annotations

from typing import Callable

from litmus.cache.manager import clear_cache as _clear_cache
from litmus.config import NOISE_TRANSFORMATION_STRATEGY, get_tier_config
from litmus.evaluate.orchestrator import evaluate_eval_set
from litmus.generate.orchestrator import generate_eval_set
from litmus.generate.sizing import calculate_size
from litmus.ingest.chunker import chunk_documents
from litmus.ingest.loader import load_directory
from litmus.llm.cost import count_tokens, estimate_llm_cost_usd
from litmus.models import BinaryQuestion, CostEstimate, EvalResults, EvalSet


def generate(
    docs_dir: str,
    llm: str = "azure/gpt-5.4",
    tier: str = "medium",
    size: int | str = "auto",
    save_path: str | None = None,
    chunking: str = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    seed: int | None = None,
    max_workers: int = 1,
    cache_dir: str | None = ".litmus_cache",
    verbose: bool = True,
) -> EvalSet:
    return generate_eval_set(
        docs_dir=docs_dir,
        llm=llm,
        tier=tier,
        size=size,
        save_path=save_path,
        chunking=chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seed=seed,
        max_workers=max_workers,
        cache_dir=cache_dir,
        verbose=verbose,
    )


def evaluate(
    eval_set: EvalSet | str,
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
    if isinstance(eval_set, str):
        eval_set = EvalSet.load(eval_set)
    return evaluate_eval_set(
        eval_set=eval_set,
        rag=rag,
        llm=llm,
        scoring=scoring,
        weights=weights,
        save_path=save_path,
        max_workers=max_workers,
        timeout=timeout,
        cache_dir=cache_dir,
        verbose=verbose,
    )


def load(path: str) -> EvalSet:
    return EvalSet.load(path)


def estimate_cost(
    docs_dir: str,
    llm: str = "azure/gpt-5.4",
    tier: str = "medium",
    size: int | str = "auto",
) -> CostEstimate:
    docs = load_directory(docs_dir)
    chunks = chunk_documents(docs)
    question_count = calculate_size(len(docs), len(chunks), tier) if size == "auto" else int(size)

    avg_chunk_tokens = (sum(c.token_count for c in chunks.values()) / len(chunks)) if chunks else 500

    # Rough per-question LLM-call estimate (generation: 1-3 calls, noise:
    # 0-2, leakage: 1), used to project total token volume. These are the
    # same "1-3 gen / 0-2 noise / 1 leakage" ranges called out in the spec's
    # Cost Estimation section.
    tier_config = get_tier_config(tier)
    noise_layer_avg = sum(len(v) for v in NOISE_TRANSFORMATION_STRATEGY.values()) / max(
        len(tier_config["noise_layers"]), 1
    )

    gen_calls_per_question = 1 + noise_layer_avg + (1 if "leakage" in tier_config["quality_checks"] else 0)
    gen_input_tokens = int(question_count * gen_calls_per_question * avg_chunk_tokens)
    gen_output_tokens = int(question_count * gen_calls_per_question * 150)

    binary_qs_per_dim = tier_config["binary_questions_per_dimension"]
    total_binary_questions = (
        sum(binary_qs_per_dim.values()) if isinstance(binary_qs_per_dim, dict) else binary_qs_per_dim * len(tier_config["scoring_dimensions"])
    )
    judge_input_tokens_per_q = avg_chunk_tokens * 2 + 300
    judge_output_tokens_per_q = total_binary_questions * 20

    eval_input_tokens = int(question_count * judge_input_tokens_per_q)
    eval_output_tokens = int(question_count * judge_output_tokens_per_q)

    gen_cost = estimate_llm_cost_usd(llm, gen_input_tokens, gen_output_tokens)
    eval_cost = estimate_llm_cost_usd(llm, eval_input_tokens, eval_output_tokens)

    minutes = question_count * 0.15

    return CostEstimate(
        generation_input_tokens=gen_input_tokens,
        generation_output_tokens=gen_output_tokens,
        generation_cost_usd=gen_cost,
        evaluation_input_tokens=eval_input_tokens,
        evaluation_output_tokens=eval_output_tokens,
        evaluation_cost_usd=eval_cost,
        total_cost_usd=gen_cost + eval_cost,
        estimated_time_minutes=minutes,
        question_count=question_count,
    )


def clear_cache(cache_dir: str = ".litmus_cache") -> None:
    _clear_cache(cache_dir)
