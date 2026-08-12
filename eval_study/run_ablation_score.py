"""Ablation Step 2: run SimpleRAG against the medium-tier FDA eval set,
score with litmus's MVP judge, compute scoring_health() -> table5.json.

Fresh SimpleRAG index built over sample1's own chunks (not reused from
Study 2's corpora) since this is a different document set entirely.
"""

from __future__ import annotations

import json

import litmus

from eval_study import config
from eval_study.simple_rag import SimpleRAG

RAG_OUTPUTS_PATH = config.ABLATION_DIR / "rag_outputs.json"
EVAL_RESULTS_PATH = config.ABLATION_DIR / "eval_results.json"
TABLE5_PATH = config.TABLES_DIR / "table5.json"


def run_rag(eval_set, rag: SimpleRAG) -> list[dict]:
    outputs = []
    for i, record in enumerate(eval_set.records):
        result = rag(record.question)
        outputs.append({"record_id": record.id, "question": record.question, **result})
        if (i + 1) % 5 == 0 or i + 1 == len(eval_set.records):
            print(f"  rag: {i + 1}/{len(eval_set.records)}")
    return outputs


def main() -> None:
    eval_set = litmus.load(str(config.ABLATION_DIR / "litmus_eval_set.json"))
    print(f"Loaded {len(eval_set.records)} records, {len(eval_set.chunks)} chunks")

    chunk_texts = [c.text for c in eval_set.chunks.values()]
    print(f"Building RAG index over {len(chunk_texts)} chunks from sample1/")
    rag = SimpleRAG(chunk_texts)

    outputs = run_rag(eval_set, rag)
    with open(RAG_OUTPUTS_PATH, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Wrote {RAG_OUTPUTS_PATH}")

    outputs_by_question = {o["question"]: o for o in outputs}

    def replay_rag(question: str) -> dict:
        o = outputs_by_question[question]
        return {"answer": o["answer"], "contexts": o["contexts"]}

    print("Scoring with litmus MVP judge...")
    results = litmus.evaluate(
        eval_set,
        rag=replay_rag,
        llm=config.FIXED_LLM,
        scoring="mvp",
        save_path=str(EVAL_RESULTS_PATH),
        cache_dir=config.CACHE_DIR,
    )

    health = results.scoring_health()
    table5 = {
        "corpus": "sample1_fda_labels",
        "tier": config.ABLATION_TIER,
        "n_records": len(eval_set.records),
        "question_type_distribution": eval_set.question_type_distribution,
        "noise_distribution": eval_set.noise_distribution,
        "scoring_health": {
            "yes_rate_per_question": health.yes_rate_per_question,
            "yes_rate_spread_per_dimension": health.yes_rate_spread_per_dimension,
            "verdict": health.verdict,
            "recommendations": health.recommendations,
            "high_correlation_pairs": health.high_correlation_pairs,
            "uncovered_failure_modes": health.uncovered_failure_modes,
        },
    }

    with open(TABLE5_PATH, "w") as f:
        json.dump(table5, f, indent=2)
    print(f"Wrote {TABLE5_PATH}")
    print(json.dumps(table5, indent=2))


if __name__ == "__main__":
    main()
