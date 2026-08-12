"""Step 5: fetch scoped Corpus B (MuSiQue) onto disk.

Samples MUSIQUE_NUM_INSTANCES QA instances from the dev set (fixed seed),
pulls the supporting+distractor paragraphs each instance is grounded in,
dedupes by title (many paragraphs repeat across instances since they're
drawn from a shared Wikipedia pool), and materializes each unique
paragraph as its own .txt file so litmus.ingest.loader.load_directory can
consume it unmodified (eval_plan.md §1.2).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil

import requests

from eval_study import config

DEV_PATH = config.SCRATCH_DIR / "musique_ans_v1.0_dev.jsonl"


def download_dev_set() -> None:
    if DEV_PATH.exists():
        print(f"Already downloaded: {DEV_PATH}")
        return
    print(f"Downloading {config.MUSIQUE_HF_URL}")
    resp = requests.get(config.MUSIQUE_HF_URL, timeout=120)
    resp.raise_for_status()
    DEV_PATH.write_bytes(resp.content)
    print(f"Wrote {DEV_PATH} ({len(resp.content)} bytes)")


def load_instances() -> list[dict]:
    with open(DEV_PATH) as f:
        return [json.loads(line) for line in f]


def sample_instances(instances: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Only answerable instances have a well-formed answer used elsewhere; keep the sample simple.
    answerable = [inst for inst in instances if inst.get("answerable", True)]
    rng.shuffle(answerable)
    return answerable[:n]


def materialize_passages(instances: list[dict]) -> list[dict]:
    if config.CORPUS_B_DOCS_DIR.exists():
        shutil.rmtree(config.CORPUS_B_DOCS_DIR)
    config.CORPUS_B_DOCS_DIR.mkdir(parents=True)

    seen_titles: dict[str, str] = {}  # title -> doc_id
    for inst in instances:
        for para in inst["paragraphs"]:
            title = para["title"]
            if title in seen_titles:
                continue
            doc_id = hashlib.sha1(title.encode()).hexdigest()[:16]
            seen_titles[title] = doc_id
            text = f"{title}\n\n{para['paragraph_text']}"
            out_path = config.CORPUS_B_DOCS_DIR / f"{doc_id}.txt"
            out_path.write_text(text)

    print(f"Materialized {len(seen_titles)} unique passages to {config.CORPUS_B_DOCS_DIR}")
    return [{"title": t, "doc_id": d} for t, d in seen_titles.items()]


def main() -> None:
    download_dev_set()
    instances = load_instances()
    print(f"Loaded {len(instances)} MuSiQue dev instances")

    sample = sample_instances(instances, config.MUSIQUE_NUM_INSTANCES, config.FIXED_SEED)
    print(f"Sampled {len(sample)} instances, seed={config.FIXED_SEED}")

    manifest = materialize_passages(sample)

    manifest_path = config.CORPUS_B_DIR / "instance_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for inst in sample:
            f.write(json.dumps({"id": inst["id"], "question": inst["question"], "answer": inst["answer"]}) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
