"""Step 11: leakage filter validation on corpus B (reuses corpus B's litmus
generation run from Step 6 -- no new generation needed per eval_plan.md §4).

litmus.generate() already runs filter_leaked_records() internally as part
of the minimal tier's quality_checks pipeline (see run_litmus_generate.py's
log: "Discarded 24 leaked questions" for corpus B). This script makes that
number an explicit, reportable Table 4 result rather than leaving it buried
in a generation log, per eval_plan.md's instruction to report discard rate
"as a standalone result, separate from the Study 2 comparison tables."

Framing (§4): this measures the filter's sensitivity on litmus's OWN
generated questions over MuSiQue passages, not a reproduction of SeedRG's
leakage-rate numbers on MuSiQue's original human-written questions.
"""

from __future__ import annotations

import json

import litmus

from eval_study import config

# From the Step 6 generation log for corpus B (run_litmus_generate.py output):
# 18 single_chunk + 18 cross_doc + 12 unanswerable + 12 adversarial candidates
# generated, of which 0 cross_doc succeeded (no bridge-entity candidates
# found) -> 18+0+12+12 = 42 candidates entered the leakage filter.
# unanswerable records are exempt from the filter (litmus.generate.leakage:
# "Unanswerable records are exempt (there's no gold answer to leak)"), so
# only the 18 single_chunk + 12 adversarial = 30 answerable candidates were
# actually checked; 24 of those were discarded, leaving the 6 answerable
# records (5 single_chunk + 1 adversarial) visible in the final eval set,
# plus the 12 exempt unanswerable records = 18 total, matching the saved set.
GENERATION_LOG_CANDIDATES = {
    "single_chunk_generated": 18,
    "cross_doc_generated": 0,
    "unanswerable_generated": 12,
    "adversarial_generated": 12,
}
GENERATION_LOG_DISCARDED = 24


def main() -> None:
    eval_set = litmus.load(str(config.CORPUS_B_DIR / "litmus_eval_set.json"))

    n_unanswerable = sum(1 for r in eval_set.records if r.unanswerable)
    n_answerable_final = len(eval_set.records) - n_unanswerable

    answerable_candidates = (
        GENERATION_LOG_CANDIDATES["single_chunk_generated"]
        + GENERATION_LOG_CANDIDATES["cross_doc_generated"]
        + GENERATION_LOG_CANDIDATES["adversarial_generated"]
    )
    discard_rate = GENERATION_LOG_DISCARDED / answerable_candidates if answerable_candidates else float("nan")

    table4 = {
        "corpus": "musique",
        "answerable_candidates_checked": answerable_candidates,
        "discarded_as_leaked": GENERATION_LOG_DISCARDED,
        "discard_rate": discard_rate,
        "unanswerable_exempt_count": n_unanswerable,
        "final_answerable_records": n_answerable_final,
        "final_eval_set_size": len(eval_set.records),
        "note": (
            "Measures filter sensitivity on litmus's own generated questions over MuSiQue "
            "passages -- not a reproduction of SeedRG's leakage-rate numbers on MuSiQue's "
            "original human-written questions."
        ),
    }

    with open(config.TABLES_DIR / "table4.json", "w") as f:
        json.dump(table4, f, indent=2)
    print(f"Wrote {config.TABLES_DIR / 'table4.json'}")
    print(json.dumps(table4, indent=2))


if __name__ == "__main__":
    main()
