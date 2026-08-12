"""Semantic versioning helpers and staleness detection for EvalSet."""

from __future__ import annotations

import hashlib
from pathlib import Path

from litmus.models import EvalSet, StaleRecord, StalenessReport


def bump_version(version: str, level: str = "minor") -> str:
    try:
        major, minor, patch = (int(p) for p in version.split("."))
    except ValueError:
        major, minor, patch = 1, 0, 0
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown version bump level: {level!r}")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_doc_hashes(docs_dir: str) -> dict[str, str]:
    from litmus.ingest.loader import SUPPORTED_EXTENSIONS

    docs_path = Path(docs_dir)
    hashes = {}
    for path in sorted(docs_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            hashes[path.stem] = _hash_file(path)
    return hashes


def check_staleness(eval_set: EvalSet, docs_dir: str) -> StalenessReport:
    current_hashes = compute_doc_hashes(docs_dir)
    original_hashes = eval_set.metadata.doc_hashes

    stale_records: list[StaleRecord] = []
    for record in eval_set.records:
        change_type = None
        for doc_id in record.source_doc_ids:
            if doc_id not in current_hashes:
                change_type = "deleted"
                break
            original = original_hashes.get(doc_id)
            if original and original != current_hashes[doc_id]:
                change_type = "modified"
                break
        if change_type:
            stale_records.append(
                StaleRecord(record_id=record.id, source_doc_ids=record.source_doc_ids, change_type=change_type)
            )

    total = len(eval_set.records)
    stale_count = len(stale_records)
    recommendations = []
    if stale_count:
        recommendations.append(f"Re-validate {stale_count} records referencing updated or removed docs")

    return StalenessReport(
        stale_records=stale_records,
        total_records=total,
        stale_count=stale_count,
        stale_ratio=(stale_count / total) if total else 0.0,
        recommendations=recommendations or ["No stale records detected."],
    )
