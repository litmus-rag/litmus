"""Step 1-2: fetch RAGTruth, join source_info+response, filter QA, map fields, stratified sample.

Produces a flat JSONL of {question, retrieved_context, generated_answer,
has_hallucination, model} per eval_plan.md §2.1, then a stratified sample
preserving the observed ~29% hallucination rate (§2.2).
"""

from __future__ import annotations

import gzip
import json
import random
import re

from eval_study import config

SOURCE_INFO_PATH = config.SCRATCH_DIR / "source_info.jsonl"
RESPONSE_PATH = config.SCRATCH_DIR / "response.jsonl"
# Gzipped: the full mapped set is ~13 MB of JSONL (~1.4 MB compressed) and is an
# intermediate artifact -- only SAMPLE_PATH feeds Study 1. Stored compressed so
# it stays committable alongside the results it backs.
MAPPED_PATH = config.RAGTRUTH_DIR / "mapped_qa_rows.jsonl.gz"
SAMPLE_PATH = config.RAGTRUTH_DIR / "sampled_rows.jsonl"

PASSAGE_MARKER_RE = re.compile(r"passage \d+:", flags=re.IGNORECASE)


def split_passages(passages: str) -> list[str]:
    """Split RAGTruth's 'passage N:' concatenated block into a list of passage texts."""
    parts = PASSAGE_MARKER_RE.split(passages)
    return [p.strip() for p in parts if p.strip()]


def load_source_info_qa() -> dict[str, dict]:
    by_source_id = {}
    with open(SOURCE_INFO_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("task_type") != "QA":
                continue
            by_source_id[row["source_id"]] = row
    return by_source_id


def build_mapped_rows() -> list[dict]:
    qa_sources = load_source_info_qa()
    mapped = []
    with open(RESPONSE_PATH) as f:
        for line in f:
            resp = json.loads(line)
            source_id = resp.get("source_id")
            src = qa_sources.get(source_id)
            if src is None:
                continue
            question = src["source_info"]["question"]
            retrieved_context = split_passages(src["source_info"]["passages"])
            mapped.append(
                {
                    "source_id": source_id,
                    "response_id": resp["id"],
                    "question": question,
                    "retrieved_context": retrieved_context,
                    "generated_answer": resp["response"],
                    "has_hallucination": len(resp.get("labels", [])) > 0,
                    "model": resp.get("model", ""),
                }
            )
    return mapped


def stratified_sample(rows: list[dict], n: int, hallucination_rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    hallucinated = [r for r in rows if r["has_hallucination"]]
    clean = [r for r in rows if not r["has_hallucination"]]
    n_hallucinated = round(n * hallucination_rate)
    n_clean = n - n_hallucinated
    rng.shuffle(hallucinated)
    rng.shuffle(clean)
    sample = hallucinated[:n_hallucinated] + clean[:n_clean]
    rng.shuffle(sample)
    return sample


def main() -> None:
    mapped = build_mapped_rows()
    print(f"Mapped QA rows: {len(mapped)}")
    n_hallucinated = sum(1 for r in mapped if r["has_hallucination"])
    print(f"Hallucinated: {n_hallucinated} ({n_hallucinated / len(mapped):.1%})")

    with gzip.open(MAPPED_PATH, "wt") as f:
        for row in mapped:
            f.write(json.dumps(row) + "\n")

    sample = stratified_sample(
        mapped,
        n=config.RAGTRUTH_SAMPLE_SIZE,
        hallucination_rate=config.RAGTRUTH_HALLUCINATION_RATE,
        seed=config.FIXED_SEED,
    )
    n_sample_hallucinated = sum(1 for r in sample if r["has_hallucination"])
    print(
        f"Sample: {len(sample)} rows, {n_sample_hallucinated} hallucinated "
        f"({n_sample_hallucinated / len(sample):.1%}), seed={config.FIXED_SEED}"
    )

    with open(SAMPLE_PATH, "w") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {MAPPED_PATH} and {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
