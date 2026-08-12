"""CSV export/import for the human review workflow."""

from __future__ import annotations

import csv
from pathlib import Path

from litmus.models import EvalSet

REVIEW_COLUMNS = [
    "id",
    "question",
    "question_clean",
    "question_type",
    "gold_answer",
    "gold_chunks_preview",
    "intent_points_summary",
    "reviewer_verdict",
    "reviewer_notes",
]


def export_review_csv(eval_set: EvalSet, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in eval_set.records:
            preview_parts = []
            for group in record.gold_chunks_text:
                preview_parts.append(" | ".join(text[:100] for text in group))
            intent_summary = "; ".join(
                f"{p.id}({'req' if p.required else 'pref'}): {p.text}" for p in record.intent_points
            )
            writer.writerow(
                {
                    "id": record.id,
                    "question": record.question,
                    "question_clean": record.question_clean,
                    "question_type": record.question_type.value,
                    "gold_answer": record.gold_answer,
                    "gold_chunks_preview": " / ".join(preview_parts),
                    "intent_points_summary": intent_summary,
                    "reviewer_verdict": "",
                    "reviewer_notes": "",
                }
            )


def apply_review_csv(eval_set: EvalSet, path: str) -> None:
    """Import reviewer corrections. Supported reviewer_verdict values:

    - "approve" / "" (blank): no change
    - "reject": remove the record from the eval set
    - "edit": apply reviewer_notes as the new gold_answer (a lightweight
      correction path - full field-level editing is out of scope for the
      CSV round-trip and should be done by editing the EvalSet JSON directly)
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by_id = {r.id: r for r in eval_set.records}
    to_remove = []
    edited_ids = []

    for row in rows:
        record_id = row.get("id", "")
        record = by_id.get(record_id)
        if record is None:
            continue
        verdict = (row.get("reviewer_verdict") or "").strip().lower()
        if verdict == "reject":
            to_remove.append(record_id)
        elif verdict == "edit":
            notes = row.get("reviewer_notes", "").strip()
            if notes:
                record.gold_answer = notes
                edited_ids.append(record_id)

    if to_remove:
        eval_set.remove_records(to_remove)
    if edited_ids:
        from litmus.version import bump_version
        from litmus.models import ChangelogEntry

        eval_set.version = bump_version(eval_set.version, "patch")
        eval_set.changelog.append(
            ChangelogEntry(
                version=eval_set.version,
                action="apply_review",
                details=f"Edited gold_answer for {len(edited_ids)} records from human review",
                record_ids=edited_ids,
            )
        )
