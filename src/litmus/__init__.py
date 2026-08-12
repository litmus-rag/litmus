"""litmus: synthetic eval-set generation and binary-decomposition scoring for RAG systems.

Public API::

    import litmus

    eval_set = litmus.generate(docs_dir="./docs", tier="medium")
    results = litmus.evaluate(eval_set, rag=my_rag_function)
    results.summary()

See README.md for the developer guide and CLAUDE.md for architecture notes.
"""

from litmus.api import clear_cache, estimate_cost, evaluate, generate, load
from litmus.models import (
    AggregateScore,
    BinaryQuestion,
    BinaryVerdict,
    Chunk,
    ComparisonReport,
    CostEstimate,
    CoverageReport,
    Difficulty,
    DimensionScore,
    EvalRecord,
    EvalResults,
    EvalSet,
    EvalSetMetadata,
    FailurePatternReport,
    FlaggedRecord,
    IntentPoint,
    IntentVerdict,
    NoiseType,
    QuestionType,
    RAGResponse,
    RecordResult,
    RetrievalReport,
    ScoringHealthReport,
    StaleRecord,
    StalenessReport,
    ValidationReport,
)

__version__ = "0.1.0"

__all__ = [
    "generate",
    "evaluate",
    "load",
    "estimate_cost",
    "clear_cache",
    "EvalSet",
    "EvalRecord",
    "EvalSetMetadata",
    "EvalResults",
    "RecordResult",
    "RAGResponse",
    "BinaryQuestion",
    "BinaryVerdict",
    "DimensionScore",
    "IntentPoint",
    "IntentVerdict",
    "Chunk",
    "QuestionType",
    "NoiseType",
    "Difficulty",
    "AggregateScore",
    "FailurePatternReport",
    "RetrievalReport",
    "ScoringHealthReport",
    "ComparisonReport",
    "CostEstimate",
    "ValidationReport",
    "FlaggedRecord",
    "CoverageReport",
    "StaleRecord",
    "StalenessReport",
]
