"""Tier presets, default weights, and diagnostic thresholds.

TIER_CONFIG mirrors the "Tier Configuration Details" section of the litmus
spec (synth file, Section on TIER_CONFIG) verbatim: question type
proportions, noise layer proportions, scoring dimensions/question counts,
retrieval metrics, quality checks, and size ranges per tier.
"""

from __future__ import annotations

from typing import Any

TIER_CONFIG: dict[str, dict[str, Any]] = {
    "minimal": {
        "question_types": {
            "single_chunk": 0.30,
            "cross_doc": 0.30,
            "unanswerable": 0.20,
            "adversarial": 0.20,
        },
        "noise_layers": {
            "vocab_mismatch": 0.40,
            "clean": 0.60,
        },
        "scoring_dimensions": ["faithfulness", "correctness", "abstention"],
        "binary_questions_per_dimension": 3,
        "retrieval_metrics": ["set_recall"],
        "quality_checks": ["leakage"],
        "size_range": (40, 80),
    },
    "medium": {
        "question_types": {
            "single_chunk": 0.20,
            "cross_doc": 0.25,
            "unanswerable": 0.15,
            "adversarial": 0.12,
            "contradiction": 0.10,
            "comparative": 0.10,
            "compound": 0.08,
        },
        "noise_layers": {
            "vocab_mismatch": 0.30,
            "vocab_mismatch+indirect": 0.20,
            "typo+fragment": 0.15,
            "clean": 0.15,
            "wrong_assumption": 0.10,
            "compound": 0.10,
        },
        "scoring_dimensions": ["faithfulness", "correctness", "abstention"],
        "binary_questions_per_dimension": 3,
        "retrieval_metrics": ["set_recall", "chunk_recall", "mrr"],
        "quality_checks": ["leakage", "eval_set_validation"],
        "size_range": (80, 200),
    },
    "exhaustive": {
        "question_types": {
            "single_chunk": 0.15,
            "cross_doc": 0.20,
            "unanswerable": 0.12,
            "adversarial": 0.10,
            "contradiction": 0.08,
            "comparative": 0.08,
            "compound": 0.08,
            "procedural": 0.07,
            "wrong_assumption": 0.07,
            "ambiguous": 0.05,
        },
        "noise_layers": {
            "vocab_mismatch": 0.25,
            "vocab_mismatch+indirect": 0.15,
            "typo+fragment": 0.12,
            "clean": 0.13,
            "wrong_assumption": 0.10,
            "compound": 0.08,
            "indirect": 0.07,
            "register_variation": 0.05,
            "fragment_only": 0.05,
        },
        "scoring_dimensions": ["faithfulness", "correctness", "abstention", "completeness", "conciseness"],
        "binary_questions_per_dimension": {
            "faithfulness": 8,
            "correctness": 5,
            "completeness": 5,
            "conciseness": 3,
            "abstention": 3,
        },
        "retrieval_metrics": ["set_recall", "chunk_recall", "mrr", "precision_at_k", "gold_chunk_ranks"],
        "quality_checks": [
            "leakage",
            "eval_set_validation_full",
            "coverage_diversity",
            "contradiction_detection",
            "staleness",
        ],
        "size_range": (200, 600),
    },
}

# Default weighted aggregation for S_overall, keyed by dimension.
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "minimal": {"faithfulness": 1 / 3, "correctness": 1 / 3, "abstention": 1 / 3},
    "medium": {"faithfulness": 1 / 3, "correctness": 1 / 3, "abstention": 1 / 3},
    "exhaustive": {
        "faithfulness": 0.35,
        "correctness": 0.25,
        "completeness": 0.20,
        "conciseness": 0.10,
        "abstention": 0.10,
    },
}

# Corpus-aware sizing table: (doc_count_upper_bound, tier) -> (low, high).
# `None` upper bound means "and above". Mirrors Section 7.5 of the spec.
SIZING_TABLE: list[tuple[int | None, dict[str, tuple[int, int]]]] = [
    (50, {"minimal": (40, 40), "medium": (80, 100), "exhaustive": (200, 200)}),
    (200, {"minimal": (50, 60), "medium": (100, 150), "exhaustive": (200, 280)}),
    (500, {"minimal": (60, 70), "medium": (150, 200), "exhaustive": (280, 400)}),
    (1000, {"minimal": (70, 80), "medium": (200, 280), "exhaustive": (350, 500)}),
    (5000, {"minimal": (80, 80), "medium": (280, 370), "exhaustive": (500, 600)}),
    (None, {"minimal": (80, 80), "medium": (370, 370), "exhaustive": (600, 600)}),
]

DOCUMENT_COVERAGE_TARGET: dict[str, tuple[float, float] | None] = {
    "minimal": None,
    "medium": (0.15, 0.30),
    "exhaustive": (0.20, 0.40),
}

NOISE_TRANSFORMATION_STRATEGY: dict[str, list[str]] = {
    "vocab_mismatch": ["vocab_mismatch"],
    "vocab_mismatch+indirect": ["vocab_mismatch", "indirect"],
    "typo+fragment": ["typo", "fragment"],
    "clean": [],
    "wrong_assumption": ["wrong_assumption"],
    "compound": ["compound"],
    "indirect": ["indirect"],
    "register_variation": ["register_variation"],
    "fragment_only": ["fragment"],
}

# Diagnostic thresholds for scoring health (Section 6.3 of the spec).
YES_RATE_SPREAD_HEALTHY = 0.20
YES_RATE_SPREAD_NEEDS_WORK = 0.10
YES_RATE_MIN_TARGET = 0.30
YES_RATE_MAX_TARGET = 0.95
MEAN_PHI_HEALTHY = 0.40
MEAN_PHI_NEEDS_WORK = 0.60
WEAK_PAIR_PHI_THRESHOLD = 0.30
WEAK_PAIR_HEALTHY_PCT = 0.50
HIGH_CORR_PHI_THRESHOLD = 0.70

FAILURE_MODES = [
    "wrong_document_retrieved",
    "partial_retrieval",
    "hallucinated_entity",
    "fabricated_statistic",
    "misattributed_statement",
    "conflated_chunks",
    "over_hedging",
    "under_hedging",
    "silent_conflict_resolution",
]

# Chunk-set discovery threshold (Section: Alternative Chunk Set Discovery).
CHUNK_DISCOVERY_SCORE_THRESHOLD = 0.8

# Leakage check: discard a generated question if the LLM answers it
# correctly with no retrieved context at all.
LEAKAGE_CHECK_ENABLED_TIERS = {"minimal", "medium", "exhaustive"}

# Contradiction detection embedding similarity pre-filter.
CONTRADICTION_SIMILARITY_THRESHOLD = 0.8

DEFAULT_LLM_MODEL = "azure/gpt-5.4"
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_GENERATION_TEMPERATURE = 0.7


def get_tier_config(tier: str) -> dict[str, Any]:
    if tier not in TIER_CONFIG:
        raise ValueError(f"Unknown tier {tier!r}. Must be one of {list(TIER_CONFIG)}")
    return TIER_CONFIG[tier]


def get_default_weights(tier: str) -> dict[str, float]:
    return dict(DEFAULT_WEIGHTS[tier])
