"""Step 6: litmus generate() on both corpora, minimal tier, fixed seed."""

from __future__ import annotations

import litmus

from eval_study import config


def main() -> None:
    print("Generating litmus eval set for Corpus A (EnterpriseRAG-Bench slice)...")
    eval_set_a = litmus.generate(
        docs_dir=str(config.CORPUS_A_DOCS_DIR),
        llm=config.FIXED_LLM,
        tier=config.LITMUS_TIER,
        seed=config.FIXED_SEED,
        save_path=str(config.CORPUS_A_DIR / "litmus_eval_set.json"),
        cache_dir=config.CACHE_DIR,
    )
    print(eval_set_a.summary())
    print("question_type_distribution:", eval_set_a.question_type_distribution)
    print("noise_distribution:", eval_set_a.noise_distribution)

    print("\nGenerating litmus eval set for Corpus B (MuSiQue passages)...")
    eval_set_b = litmus.generate(
        docs_dir=str(config.CORPUS_B_DOCS_DIR),
        llm=config.FIXED_LLM,
        tier=config.LITMUS_TIER,
        seed=config.FIXED_SEED,
        save_path=str(config.CORPUS_B_DIR / "litmus_eval_set.json"),
        cache_dir=config.CACHE_DIR,
    )
    print(eval_set_b.summary())
    print("question_type_distribution:", eval_set_b.question_type_distribution)
    print("noise_distribution:", eval_set_b.noise_distribution)


if __name__ == "__main__":
    main()
