"""Step 3-4: thin wrapper around litmus's run_judge for RAGTruth rows, plus kappa/P/R/F1.

Runs only F1/F2/F3 (faithfulness) from MVP_QUESTIONS against the mapped
RAGTruth rows, bypassing the normal EvalRecord/RAGResponse flow since
RAGTruth isn't a litmus-native eval set (eval_plan.md §2.3, Step 3).
"""

from __future__ import annotations

import json

from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from litmus.evaluate.answer_scorer import MVP_QUESTIONS
from litmus.llm.client import LLMClient

from eval_study import config

FAITHFULNESS_QUESTIONS = [q for q in MVP_QUESTIONS if q.dimension == "faithfulness"]
assert [q.id for q in FAITHFULNESS_QUESTIONS] == ["F1", "F2", "F3"]

SAMPLE_PATH = config.RAGTRUTH_DIR / "sampled_rows.jsonl"
VERDICTS_PATH = config.RAGTRUTH_DIR / "judge_verdicts.jsonl"
TABLE1_PATH = config.TABLES_DIR / "table1.json"


def run_judge_on_row(row: dict, client: LLMClient) -> dict:
    """Run F1/F2/F3 on one mapped RAGTruth row. gold_answer is unavailable (empty string);
    F1-F3 don't require it. is_unanswerable is always False for RAGTruth QA rows."""
    from litmus.evaluate.answer_scorer import run_judge

    verdicts = run_judge(
        question=row["question"],
        retrieved_context=row["retrieved_context"],
        generated_answer=row["generated_answer"],
        gold_answer="",
        is_unanswerable=False,
        questions=FAITHFULNESS_QUESTIONS,
        client=client,
    )
    judge_says_hallucinated = not all(verdicts[q.id].verdict for q in FAITHFULNESS_QUESTIONS)
    return {
        "source_id": row["source_id"],
        "response_id": row["response_id"],
        "model": row["model"],
        "has_hallucination": row["has_hallucination"],
        "judge_says_hallucinated": judge_says_hallucinated,
        "verdicts": {qid: {"verdict": v.verdict, "reason": v.reason} for qid, v in verdicts.items()},
    }


def run_all(rows: list[dict], client: LLMClient) -> list[dict]:
    results = []
    for i, row in enumerate(rows):
        result = run_judge_on_row(row, client)
        results.append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"  judged {i + 1}/{len(rows)}")
    return results


def compute_kappa_prf(y_true: list[bool], y_pred: list[bool]) -> dict:
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        kappa = float("nan")
    else:
        kappa = cohen_kappa_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=True, average="binary", zero_division=0
    )
    raw_agreement = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "raw_agreement": raw_agreement,
        "n": len(y_true),
    }


def compute_table1(judged: list[dict]) -> dict:
    y_true = [r["has_hallucination"] for r in judged]
    y_pred_combined = [r["judge_says_hallucinated"] for r in judged]
    overall = compute_kappa_prf(y_true, y_pred_combined)

    per_question = {}
    for q in FAITHFULNESS_QUESTIONS:
        # Per-question judge verdict "hallucinated" = the question answered "no".
        y_pred_q = [not r["verdicts"][q.id]["verdict"] for r in judged]
        per_question[q.id] = compute_kappa_prf(y_true, y_pred_q)

    return {
        "overall": overall,
        "per_question": per_question,
        "hallucination_rate_ground_truth": sum(y_true) / len(y_true),
        "hallucination_rate_judge": sum(y_pred_combined) / len(y_pred_combined),
    }


def main() -> None:
    rows = [json.loads(line) for line in open(SAMPLE_PATH)]
    print(f"Loaded {len(rows)} sampled rows")

    client = LLMClient(model=config.FIXED_LLM)
    judged = run_all(rows, client)

    with open(VERDICTS_PATH, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {VERDICTS_PATH}")

    table1 = compute_table1(judged)
    with open(TABLE1_PATH, "w") as f:
        json.dump(table1, f, indent=2)
    print(f"Wrote {TABLE1_PATH}")
    print(json.dumps(table1, indent=2))


if __name__ == "__main__":
    main()
