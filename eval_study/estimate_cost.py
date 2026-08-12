"""Step 5: cost estimation for both corpora, logged before generation runs."""

from __future__ import annotations

import json

import litmus

from eval_study import config

OUT_PATH = config.RESULTS_DIR / "cost_estimates.json"


def main() -> None:
    est_a = litmus.estimate_cost(docs_dir=str(config.CORPUS_A_DOCS_DIR), llm=config.FIXED_LLM, tier=config.LITMUS_TIER)
    est_b = litmus.estimate_cost(docs_dir=str(config.CORPUS_B_DOCS_DIR), llm=config.FIXED_LLM, tier=config.LITMUS_TIER)
    print("Corpus A estimate:\n", est_a)
    print("Corpus B estimate:\n", est_b)
    with open(OUT_PATH, "w") as f:
        json.dump({"corpus_a": est_a.to_dict(), "corpus_b": est_b.to_dict()}, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
