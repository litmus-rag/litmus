"""Step 8: stand up the simple RAG pipeline once per corpus, run it against
that corpus's litmus- and Ragas-generated eval sets.

Per eval_plan.md §0.5/§3.2, the RAG system is held constant within a corpus
(litmus-A and ragas-A share one SimpleRAG instance built from corpus A's
chunk pool; litmus-B and ragas-B share a separate instance from corpus B's
chunk pool) so only the eval set varies, not the RAG or the retrieval index.
"""

from __future__ import annotations

import json

import litmus

from eval_study import config
from eval_study.simple_rag import SimpleRAG

RAG_OUTPUTS_SUFFIX = "_rag_outputs.json"


def run_litmus_set(rag: SimpleRAG, eval_set) -> list[dict]:
    outputs = []
    for i, record in enumerate(eval_set.records):
        result = rag(record.question)
        outputs.append({"record_id": record.id, "question": record.question, **result})
        if (i + 1) % 5 == 0 or i + 1 == len(eval_set.records):
            print(f"  litmus set: {i + 1}/{len(eval_set.records)}")
    return outputs


def run_ragas_rows(rag: SimpleRAG, rows: list[dict]) -> list[dict]:
    outputs = []
    for i, row in enumerate(rows):
        result = rag(row["question"])
        outputs.append({"question": row["question"], "ground_truth": row["ground_truth"], **result})
        if (i + 1) % 5 == 0 or i + 1 == len(rows):
            print(f"  ragas set: {i + 1}/{len(rows)}")
    return outputs


def run_for_corpus(corpus_dir) -> None:
    litmus_eval_set = litmus.load(str(corpus_dir / "litmus_eval_set.json"))
    ragas_rows = json.load(open(corpus_dir / "ragas_eval_set.json"))

    chunk_texts = [c.text for c in litmus_eval_set.chunks.values()]
    print(f"{corpus_dir.name}: building RAG index over {len(chunk_texts)} chunks")
    rag = SimpleRAG(chunk_texts)

    print(f"{corpus_dir.name}: running RAG against litmus eval set ({len(litmus_eval_set.records)} records)")
    litmus_outputs = run_litmus_set(rag, litmus_eval_set)
    with open(corpus_dir / f"litmus{RAG_OUTPUTS_SUFFIX}", "w") as f:
        json.dump(litmus_outputs, f, indent=2)

    print(f"{corpus_dir.name}: running RAG against ragas eval set ({len(ragas_rows)} rows)")
    ragas_outputs = run_ragas_rows(rag, ragas_rows)
    with open(corpus_dir / f"ragas{RAG_OUTPUTS_SUFFIX}", "w") as f:
        json.dump(ragas_outputs, f, indent=2)


def main() -> None:
    run_for_corpus(config.CORPUS_A_DIR)
    run_for_corpus(config.CORPUS_B_DIR)


if __name__ == "__main__":
    main()
