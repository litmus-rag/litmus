"""pandas DataFrame export for EvalResults (exhaustive tier convenience, but
works for any tier)."""

from __future__ import annotations

from litmus.models import EvalResults


def results_to_dataframe(results: EvalResults):
    import pandas as pd

    rows = []
    for r in results.records:
        rows.append(
            {
                "record_id": r.record_id,
                "question": r.question,
                "question_type": r.question_type.value,
                "noise_profile": "+".join(n.value for n in r.noise_profile),
                "difficulty": r.difficulty.value,
                "rag_answer": r.rag_answer,
                "rag_error": r.rag_error,
                "chunk_recall": r.chunk_recall,
                "set_recall": r.set_recall,
                "mrr": r.mrr,
                "precision_at_k": r.precision_at_k,
                "faithfulness": r.faithfulness.score,
                "correctness": r.correctness.score,
                "abstention": r.abstention.score,
                "completeness": r.completeness.score if r.completeness else None,
                "conciseness": r.conciseness.score if r.conciseness else None,
                "overall_score": r.overall_score,
                "intent_coverage": r.intent_coverage,
                "domain_tags": ",".join(r.domain_tags),
            }
        )
    return pd.DataFrame(rows)
