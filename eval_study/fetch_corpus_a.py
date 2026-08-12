"""Step 5: fetch scoped Corpus A (EnterpriseRAG-Bench slice) onto disk.

Downloads one small source-type slice's document zip from the v1.0.0
release, extracts, and caps to ENTERPRISERAG_MAX_DOCS documents. Litmus and
Ragas generate their own questions from these documents in Step 6/7 — we
don't need EnterpriseRAG-Bench's own questions.jsonl for that; the taxonomy
comparison in eval_plan.md §1.1 is conceptual (comparing question-type
category names), not a literal reuse of their gold Q&A pairs.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

from eval_study import config

SLICE_URL = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/"
    f"v1.0.0/{config.ENTERPRISERAG_SOURCE_TYPE}_slice_0001.zip"
)
ZIP_PATH = config.SCRATCH_DIR / f"{config.ENTERPRISERAG_SOURCE_TYPE}_slice_0001.zip"


def download_slice() -> None:
    if ZIP_PATH.exists():
        print(f"Already downloaded: {ZIP_PATH}")
        return
    print(f"Downloading {SLICE_URL}")
    resp = requests.get(SLICE_URL, timeout=120)
    resp.raise_for_status()
    ZIP_PATH.write_bytes(resp.content)
    print(f"Wrote {ZIP_PATH} ({len(resp.content)} bytes)")


def extract_docs(max_docs: int) -> list[Path]:
    if config.CORPUS_A_DOCS_DIR.exists():
        shutil.rmtree(config.CORPUS_A_DOCS_DIR)
    config.CORPUS_A_DOCS_DIR.mkdir(parents=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
        selected = names[:max_docs]
        for name in selected:
            data = zf.read(name)
            out_path = config.CORPUS_A_DOCS_DIR / Path(name).name
            out_path.write_bytes(data)

    written = sorted(config.CORPUS_A_DOCS_DIR.glob("*.txt"))
    print(f"Extracted {len(written)} docs to {config.CORPUS_A_DOCS_DIR}")
    return written


def main() -> None:
    download_slice()
    extract_docs(config.ENTERPRISERAG_MAX_DOCS)


if __name__ == "__main__":
    main()
