"""Step 14: recompute yes-rate spreads at 1x (v3) and 2x (v4) corpus size
into a single archived artifact, backing the paper's Table 6.

Pure math over already-scored result files -- no LLM calls, no cost.
"""

from __future__ import annotations

import json

from eval_study import config

DIMS = ["faithfulness", "correctness", "abstention"]


def spreads(path) -> dict:
    recs = json.load(open(path))["records"]
    out = {}
    for dim in DIMS:
        yes, tot = {}, {}
        for r in recs:
            ds = r.get(dim)
            if not ds:
                continue
            for qid, v in ds["verdicts"].items():
                tot[qid] = tot.get(qid, 0) + 1
                yes[qid] = yes.get(qid, 0) + (1 if v["verdict"] else 0)
        rates = {q: yes[q] / tot[q] for q in tot}
        out[dim] = max(rates.values()) - min(rates.values()) if rates else None
    return {"n": len(recs), "yes_rate_spread_per_dimension": out}


def main() -> None:
    table = {}
    for corpus, d in [("corpus_a", config.CORPUS_A_DIR), ("corpus_b", config.CORPUS_B_DIR)]:
        table[corpus] = {
            f"{fw}_{size}": spreads(d / f"{fw}_eval_results_{ver}.json")
            for fw in ("litmus", "ragas")
            for size, ver in (("1x", "v3"), ("2x", "v4"))
        }
    table["note"] = (
        "Yes-rate spread at 1x (v3) and 2x (v4) corpus size. Bridge-entity "
        "candidate counts in the paper's Table 6 are from generation logs "
        "and are NOT recomputed here."
    )

    out_path = config.TABLES_DIR / "table6.json"
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
