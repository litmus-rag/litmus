"""Steps 9-10: wrap Ragas's output as a synthetic litmus EvalSet, score both
tools' eval sets with litmus's MVP judge (same model, held constant per
eval_plan.md §3.2), compute scoring_health() and question-type/noise
distributions -> Table 2/3 data.

Ragas doesn't provide gold_chunk_ids the way litmus does, so retrieval
metrics for the Ragas set are not directly comparable to litmus's -- we
only compare on judge-scored dimensions (faithfulness/correctness/
abstention) and scoring-health diagnostics, which is what eval_plan.md's
Table 3 actually asks for.
"""

from __future__ import annotations

import json

import litmus
from litmus.models import Difficulty, EvalRecord, EvalSet, EvalSetMetadata, QuestionType

from eval_study import config
from eval_study.run_rag import RAG_OUTPUTS_SUFFIX


def ragas_rows_to_eval_set(rows: list[dict], source_corpus_dir, tier: str = config.LITMUS_TIER) -> EvalSet:
    records = []
    for i, row in enumerate(rows):
        records.append(
            EvalRecord(
                id=f"ragas_{i}",
                question=row["question"],
                question_clean=row["question"],
                question_type=QuestionType.SINGLE_CHUNK,  # ragas doesn't label question types the same way
                noise_profile=[],
                difficulty=Difficulty.MEDIUM,
                gold_answer=row["ground_truth"],
                gold_chunk_ids=[],
                gold_chunks_text=[row["contexts"]] if row["contexts"] else [],
                unanswerable=False,
                domain_tags=[],
                source_doc_ids=[],
                scoring_notes=row.get("synthesizer_name", ""),
            )
        )
    return EvalSet(
        records=records,
        chunks={},
        metadata=EvalSetMetadata(docs_dir=str(source_corpus_dir), llm=config.FIXED_LLM),
        tier=tier,
    )


def score_precomputed(eval_set: EvalSet, outputs: list[dict], save_path) -> "litmus.EvalResults":
    """Score an eval set whose RAG outputs were already computed (Step 8),
    by handing evaluate() a callable that just replays the cached answer for
    each question rather than re-running the RAG live."""
    outputs_by_question = {o["question"]: o for o in outputs}

    def replay_rag(question: str) -> dict:
        o = outputs_by_question[question]
        return {"answer": o["answer"], "contexts": o["contexts"]}

    return litmus.evaluate(
        eval_set,
        rag=replay_rag,
        llm=config.FIXED_LLM,
        scoring="mvp",
        save_path=str(save_path) if save_path else None,
        cache_dir=config.CACHE_DIR,
    )


def score_corpus(corpus_dir) -> dict:
    litmus_eval_set = litmus.load(str(corpus_dir / "litmus_eval_set.json"))
    ragas_rows = json.load(open(corpus_dir / "ragas_eval_set.json"))
    ragas_eval_set = ragas_rows_to_eval_set(ragas_rows, corpus_dir)

    litmus_outputs = json.load(open(corpus_dir / f"litmus{RAG_OUTPUTS_SUFFIX}"))
    ragas_outputs = json.load(open(corpus_dir / f"ragas{RAG_OUTPUTS_SUFFIX}"))

    print(f"{corpus_dir.name}: scoring litmus eval set with MVP judge...")
    litmus_results = score_precomputed(litmus_eval_set, litmus_outputs, corpus_dir / "litmus_eval_results.json")
    print(f"{corpus_dir.name}: scoring ragas eval set with MVP judge...")
    ragas_results = score_precomputed(ragas_eval_set, ragas_outputs, corpus_dir / "ragas_eval_results.json")

    litmus_health = litmus_results.scoring_health()
    ragas_health = ragas_results.scoring_health()

    table2_block = {
        "litmus_question_type_distribution": litmus_eval_set.question_type_distribution,
        "litmus_noise_distribution": litmus_eval_set.noise_distribution,
        "ragas_synthesizer_distribution": _count(r.scoring_notes for r in ragas_eval_set.records),
        "litmus_n": len(litmus_eval_set.records),
        "ragas_n": len(ragas_eval_set.records),
    }

    table3_block = {
        "litmus_scoring_health": _health_to_dict(litmus_health),
        "ragas_scoring_health": _health_to_dict(ragas_health),
    }

    return {"table2": table2_block, "table3": table3_block}


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _health_to_dict(health) -> dict:
    return {
        "yes_rate_per_question": health.yes_rate_per_question,
        "yes_rate_spread_per_dimension": health.yes_rate_spread_per_dimension,
        "verdict": health.verdict,
        "recommendations": health.recommendations,
        "high_correlation_pairs": health.high_correlation_pairs,
        "uncovered_failure_modes": health.uncovered_failure_modes,
    }


def main() -> None:
    table2 = {}
    table3 = {}
    for corpus_dir, name in [(config.CORPUS_A_DIR, "corpus_a"), (config.CORPUS_B_DIR, "corpus_b")]:
        blocks = score_corpus(corpus_dir)
        table2[name] = blocks["table2"]
        table3[name] = blocks["table3"]

    with open(config.TABLES_DIR / "table2.json", "w") as f:
        json.dump(table2, f, indent=2)
    with open(config.TABLES_DIR / "table3.json", "w") as f:
        json.dump(table3, f, indent=2)
    print(f"Wrote {config.TABLES_DIR / 'table2.json'} and {config.TABLES_DIR / 'table3.json'}")
    print(json.dumps({"table2": table2, "table3": table3}, indent=2))


if __name__ == "__main__":
    main()
