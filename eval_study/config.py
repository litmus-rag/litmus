"""Fixed model, seed, scoped sizes, and paths for the litmus eval study.

Scoped-down sizes for fast iteration (see eval_plan.md and the plan this
package implements). Bump these constants to scale up later.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = REPO_ROOT / "conf"

# Single model used for litmus generation, litmus judging, and Ragas generation
# (per eval_plan.md §0.5 — model consistency across all three roles).
FIXED_LLM = "azure/gpt-5.4"

FIXED_SEED = 20260723

RESULTS_DIR = REPO_ROOT / "results"
RAGTRUTH_DIR = RESULTS_DIR / "ragtruth_calibration"
CORPUS_A_DIR = RESULTS_DIR / "corpus_a_enterpriserag"
CORPUS_B_DIR = RESULTS_DIR / "corpus_b_musique"
TABLES_DIR = RESULTS_DIR / "tables"

SCRATCH_DIR = REPO_ROOT / "eval_study" / ".scratch"
CORPUS_A_DOCS_DIR = SCRATCH_DIR / "corpus_a_docs"
CORPUS_B_DOCS_DIR = SCRATCH_DIR / "corpus_b_docs"

CACHE_DIR = str(REPO_ROOT / ".litmus_cache")

# --- Study 1: RAGTruth judge calibration ---
RAGTRUTH_SAMPLE_SIZE = 110
RAGTRUTH_HALLUCINATION_RATE = 0.29  # observed rate on the full QA subset; preserved by stratified sampling

RAGTRUTH_RAW_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"

# --- Study 2/3: corpora ---
# Corpus A: EnterpriseRAG-Bench, one small source-type slice.
ENTERPRISERAG_SOURCE_TYPE = "github"
ENTERPRISERAG_MAX_DOCS = 100  # bumped 2x from the initial 50-doc scoped run
ENTERPRISERAG_RAW_BASE = "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main"

# Corpus B: MuSiQue dev set, a handful of QA instances' worth of passages.
MUSIQUE_NUM_INSTANCES = 40  # bumped 2x from the initial 20-instance scoped run
MUSIQUE_HF_URL = (
    "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl"
)

LITMUS_TIER = "minimal"

# Ablation: medium-tier stress test on the FDA drug-label corpus (sample1/),
# litmus-only (no Ragas) -- checking whether a harder corpus + tier produces
# discriminating (non-saturated) scoring_health results.
ABLATION_DOCS_DIR = REPO_ROOT / "sample1"
ABLATION_DIR = RESULTS_DIR / "ablation_medium_fda"
ABLATION_TIER = "medium"
ABLATION_SIZE = 50

for _d in (ABLATION_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

for _d in (RESULTS_DIR, RAGTRUTH_DIR, CORPUS_A_DIR, CORPUS_B_DIR, TABLES_DIR, SCRATCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)
