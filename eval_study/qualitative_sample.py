"""Step 12: pull cross-doc examples from each tool's Step 6/7 output per
corpus, for manual qualitative review (eval_plan.md §3.2 item 3, §5 item 5).

This is illustrative, not a statistical claim (§3.2's "small-N, not a large
study" instruction) -- writes a plain text file for a human to read and
annotate, not an automated score.
"""

from __future__ import annotations

import json

import litmus

from eval_study import config

N_EXAMPLES_PER_TOOL = 10


def litmus_cross_doc_examples(corpus_dir, n: int) -> list[dict]:
    eval_set = litmus.load(str(corpus_dir / "litmus_eval_set.json"))
    cross_doc = [r for r in eval_set.records if r.question_type.value == "cross_doc"]
    out = []
    for r in cross_doc[:n]:
        out.append(
            {
                "question": r.question,
                "gold_answer": r.gold_answer,
                "source_doc_ids": r.source_doc_ids,
                "requires_synthesis": r.requires_synthesis,
            }
        )
    return out


def ragas_multihop_examples(corpus_dir, n: int) -> list[dict]:
    rows = json.load(open(corpus_dir / "ragas_eval_set.json"))
    multihop = [r for r in rows if "multi_hop" in r.get("synthesizer_name", "")]
    out = []
    for r in multihop[:n]:
        out.append(
            {
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "num_contexts": len(r["contexts"]),
                "synthesizer_name": r["synthesizer_name"],
            }
        )
    return out


def write_review_file(corpus_name: str, corpus_dir, out_path) -> None:
    litmus_examples = litmus_cross_doc_examples(corpus_dir, N_EXAMPLES_PER_TOOL)
    ragas_examples = ragas_multihop_examples(corpus_dir, N_EXAMPLES_PER_TOOL)

    lines = [f"# Cross-doc / multi-hop qualitative review: {corpus_name}", ""]
    lines.append(f"## litmus cross_doc questions ({len(litmus_examples)} of type cross_doc found)")
    lines.append("")
    if not litmus_examples:
        lines.append(
            "(none -- litmus's bridge-entity scanner found 0 shared-entity candidates in this "
            "corpus, so no cross_doc questions were generated; see generation log)"
        )
    for i, ex in enumerate(litmus_examples, 1):
        lines.append(f"{i}. Q: {ex['question']}")
        lines.append(f"   A: {ex['gold_answer']}")
        lines.append(f"   source_doc_ids: {ex['source_doc_ids']}  requires_synthesis: {ex['requires_synthesis']}")
        lines.append("   REVIEW: [genuine cross-doc / pseudo multi-hop?] ___")
        lines.append("")

    lines.append(f"## Ragas multi_hop_specific_query_synthesizer questions ({len(ragas_examples)} found)")
    lines.append("")
    for i, ex in enumerate(ragas_examples, 1):
        lines.append(f"{i}. Q: {ex['question']}")
        lines.append(f"   A: {ex['ground_truth']}")
        lines.append(f"   num_contexts: {ex['num_contexts']}  synthesizer: {ex['synthesizer_name']}")
        lines.append("   REVIEW: [genuine cross-doc / pseudo multi-hop -- answerable from one chunk?] ___")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    write_review_file("Corpus A (EnterpriseRAG-Bench)", config.CORPUS_A_DIR, config.RESULTS_DIR / "qualitative_review_corpus_a.md")
    write_review_file("Corpus B (MuSiQue)", config.CORPUS_B_DIR, config.RESULTS_DIR / "qualitative_review_corpus_b.md")


if __name__ == "__main__":
    main()
