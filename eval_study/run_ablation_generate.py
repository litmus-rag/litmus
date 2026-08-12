"""Ablation: medium-tier litmus generation on the FDA drug-label corpus
(sample1/), litmus-only (no Ragas comparison) -- checking whether a harder
corpus + tier produces discriminating scoring_health results, since Study
2's scoped-down corpora saturated to "broken" on both litmus's and Ragas's
eval sets (see results/tables/table3.json).
"""

from __future__ import annotations

import litmus

from eval_study import config


def main() -> None:
    print(f"Generating medium-tier litmus eval set for {config.ABLATION_DOCS_DIR} (size={config.ABLATION_SIZE})...")
    eval_set = litmus.generate(
        docs_dir=str(config.ABLATION_DOCS_DIR),
        llm=config.FIXED_LLM,
        tier=config.ABLATION_TIER,
        size=config.ABLATION_SIZE,
        seed=config.FIXED_SEED,
        save_path=str(config.ABLATION_DIR / "litmus_eval_set.json"),
        cache_dir=config.CACHE_DIR,
    )
    print(eval_set.summary())
    print("question_type_distribution:", eval_set.question_type_distribution)
    print("noise_distribution:", eval_set.noise_distribution)


if __name__ == "__main__":
    main()
