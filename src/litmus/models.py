"""All dataclasses and enums for litmus.

Data models are plain ``dataclasses`` (not pydantic) so that ``EvalSet`` and
``EvalResults`` can carry behavior (``save``, ``validate``, ``summary``, ...)
without fighting pydantic's mutation model. JSON (de)serialization is
hand-rolled in this module via ``to_dict``/``from_dict`` pairs.

Methods on ``EvalSet``/``EvalResults`` that need heavier logic (validation,
staleness, diagnostics, review export) import their implementation module
lazily inside the method body to avoid circular imports, since those modules
import ``models`` at top level.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    SINGLE_CHUNK = "single_chunk"
    CROSS_DOC = "cross_doc"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"
    CONTRADICTION = "contradiction"
    COMPARATIVE = "comparative"
    COMPOUND = "compound"
    PROCEDURAL = "procedural"
    WRONG_ASSUMPTION = "wrong_assumption"
    AMBIGUOUS = "ambiguous"


class NoiseType(str, Enum):
    CLEAN = "clean"
    VOCAB_MISMATCH = "vocab_mismatch"
    INDIRECT = "indirect"
    TYPO = "typo"
    FRAGMENT = "fragment"
    WRONG_ASSUMPTION = "wrong_assumption"
    COMPOUND = "compound"
    REGISTER_VARIATION = "register_variation"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_enum_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _enum_safe(v) for k, v in value.items()}
    return value


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Core small models
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class IntentPoint:
    id: str
    text: str
    required: bool
    source_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentPoint":
        return cls(**data)


@dataclass
class BinaryQuestion:
    id: str
    dimension: str
    text: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BinaryQuestion":
        return cls(**data)


@dataclass
class BinaryVerdict:
    question_id: str
    verdict: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BinaryVerdict":
        return cls(**data)


@dataclass
class DimensionScore:
    score: float
    verdicts: dict[str, BinaryVerdict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdicts": {k: v.to_dict() for k, v in self.verdicts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DimensionScore":
        return cls(
            score=data["score"],
            verdicts={k: BinaryVerdict.from_dict(v) for k, v in data.get("verdicts", {}).items()},
        )


@dataclass
class IntentVerdict:
    intent_point_id: str
    covered: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentVerdict":
        return cls(**data)


# ---------------------------------------------------------------------------
# RAG interface
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    answer: str
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGResponse":
        return cls(**data)


def normalize_rag_response(raw: Any) -> RAGResponse:
    """Coerce a user RAG callable's return value into a RAGResponse.

    Accepts dict ({"answer": ..., "contexts": [...]}), tuple/list
    (answer, contexts), or a RAGResponse already.
    """
    if isinstance(raw, RAGResponse):
        return raw
    if isinstance(raw, dict):
        return RAGResponse(answer=raw.get("answer", ""), contexts=list(raw.get("contexts", [])))
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        answer, contexts = raw
        return RAGResponse(answer=answer, contexts=list(contexts))
    raise TypeError(
        f"RAG callable must return dict, tuple/list of (answer, contexts), or RAGResponse, got {type(raw)!r}"
    )


# ---------------------------------------------------------------------------
# EvalRecord
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    id: str
    question: str
    question_clean: str
    question_type: QuestionType
    noise_profile: list[NoiseType]
    difficulty: Difficulty
    gold_answer: str
    gold_chunk_ids: list[list[str]] = field(default_factory=list)
    gold_chunks_text: list[list[str]] = field(default_factory=list)
    intent_points: list[IntentPoint] = field(default_factory=list)
    unanswerable: bool = False
    requires_synthesis: bool = False
    domain_tags: list[str] = field(default_factory=list)
    source_doc_ids: list[str] = field(default_factory=list)
    scoring_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "question_clean": self.question_clean,
            "question_type": self.question_type.value,
            "noise_profile": [n.value for n in self.noise_profile],
            "difficulty": self.difficulty.value,
            "gold_answer": self.gold_answer,
            "gold_chunk_ids": self.gold_chunk_ids,
            "gold_chunks_text": self.gold_chunks_text,
            "intent_points": [p.to_dict() for p in self.intent_points],
            "unanswerable": self.unanswerable,
            "requires_synthesis": self.requires_synthesis,
            "domain_tags": self.domain_tags,
            "source_doc_ids": self.source_doc_ids,
            "scoring_notes": self.scoring_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRecord":
        return cls(
            id=data["id"],
            question=data["question"],
            question_clean=data["question_clean"],
            question_type=QuestionType(data["question_type"]),
            noise_profile=[NoiseType(n) for n in data.get("noise_profile", [])],
            difficulty=Difficulty(data["difficulty"]),
            gold_answer=data["gold_answer"],
            gold_chunk_ids=data.get("gold_chunk_ids", []),
            gold_chunks_text=data.get("gold_chunks_text", []),
            intent_points=[IntentPoint.from_dict(p) for p in data.get("intent_points", [])],
            unanswerable=data.get("unanswerable", False),
            requires_synthesis=data.get("requires_synthesis", False),
            domain_tags=data.get("domain_tags", []),
            source_doc_ids=data.get("source_doc_ids", []),
            scoring_notes=data.get("scoring_notes", ""),
        )


# ---------------------------------------------------------------------------
# EvalSet
# ---------------------------------------------------------------------------


@dataclass
class ChangelogEntry:
    version: str
    action: str
    details: str
    record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangelogEntry":
        return cls(**data)


@dataclass
class EvalSetMetadata:
    docs_dir: str = ""
    llm: str = ""
    num_source_docs: int = 0
    num_chunks: int = 0
    generated_at: str = ""
    chunking: str = "auto"
    chunk_size: int = 512
    chunk_overlap: int = 64
    seed: int | None = None
    doc_hashes: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSetMetadata":
        return cls(**data)


@dataclass
class EvalSet:
    records: list[EvalRecord]
    chunks: dict[str, Chunk]
    metadata: EvalSetMetadata
    tier: str
    version: str = "1.0.0"
    changelog: list[ChangelogEntry] = field(default_factory=list)

    # -- persistence --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tier": self.tier,
            "metadata": self.metadata.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "chunks": {k: v.to_dict() for k, v in self.chunks.items()},
            "changelog": [c.to_dict() for c in self.changelog],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSet":
        return cls(
            version=data.get("version", "1.0.0"),
            tier=data["tier"],
            metadata=EvalSetMetadata.from_dict(data.get("metadata", {})),
            records=[EvalRecord.from_dict(r) for r in data.get("records", [])],
            chunks={k: Chunk.from_dict(v) for k, v in data.get("chunks", {}).items()},
            changelog=[ChangelogEntry.from_dict(c) for c in data.get("changelog", [])],
        )

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "EvalSet":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- quality checks -------------------------------------------------

    def validate(self):
        from litmus.generate.quality import validate_eval_set

        return validate_eval_set(self)

    def check_staleness(self, docs_dir: str):
        from litmus.version import check_staleness

        return check_staleness(self, docs_dir)

    # -- human review ----------------------------------------------------

    def to_review_csv(self, path: str) -> None:
        from litmus.report.review_export import export_review_csv

        export_review_csv(self, path)

    def apply_review(self, path: str) -> None:
        from litmus.report.review_export import apply_review_csv

        apply_review_csv(self, path)

    # -- stats ------------------------------------------------------------

    def summary(self) -> None:
        from litmus.report.summary import print_eval_set_summary

        print_eval_set_summary(self)

    def filter(self, **kwargs) -> "EvalSet":
        records = self.records
        for key, value in kwargs.items():
            if key == "question_type":
                want = value if isinstance(value, (list, tuple, set)) else [value]
                want = {QuestionType(w) if not isinstance(w, QuestionType) else w for w in want}
                records = [r for r in records if r.question_type in want]
            elif key == "difficulty":
                want = value if isinstance(value, (list, tuple, set)) else [value]
                want = {Difficulty(w) if not isinstance(w, Difficulty) else w for w in want}
                records = [r for r in records if r.difficulty in want]
            elif key == "noise_type":
                want = value if isinstance(value, (list, tuple, set)) else [value]
                want = {NoiseType(w) if not isinstance(w, NoiseType) else w for w in want}
                records = [r for r in records if want & set(r.noise_profile)]
            elif key == "domain_tag":
                records = [r for r in records if value in r.domain_tags]
            elif key == "unanswerable":
                records = [r for r in records if r.unanswerable == value]
            else:
                raise ValueError(f"Unknown filter key: {key}")
        used_chunk_ids = {cid for r in records for group in r.gold_chunk_ids for cid in group}
        chunks = {k: v for k, v in self.chunks.items() if k in used_chunk_ids}
        return EvalSet(
            records=records,
            chunks=chunks,
            metadata=self.metadata,
            tier=self.tier,
            version=self.version,
            changelog=list(self.changelog),
        )

    @property
    def question_type_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.question_type.value] = out.get(r.question_type.value, 0) + 1
        return out

    @property
    def noise_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            key = "+".join(sorted(n.value for n in r.noise_profile)) or "clean"
            out[key] = out.get(key, 0) + 1
        return out

    @property
    def document_coverage(self) -> float:
        all_docs = {c.doc_id for c in self.chunks.values()}
        if not all_docs:
            return 0.0
        covered = {doc_id for r in self.records for doc_id in r.source_doc_ids}
        return len(covered & all_docs) / len(all_docs)

    @property
    def domain_coverage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            for tag in r.domain_tags:
                out[tag] = out.get(tag, 0) + 1
        return out

    # -- versioning -----------------------------------------------------

    def add_records(self, records: list[EvalRecord]) -> None:
        from litmus.version import bump_version

        self.records.extend(records)
        self.version = bump_version(self.version, "minor")
        self.changelog.append(
            ChangelogEntry(
                version=self.version,
                action="add_records",
                details=f"Added {len(records)} records",
                record_ids=[r.id for r in records],
            )
        )

    def remove_records(self, ids: list[str]) -> None:
        from litmus.version import bump_version

        id_set = set(ids)
        self.records = [r for r in self.records if r.id not in id_set]
        self.version = bump_version(self.version, "minor")
        self.changelog.append(
            ChangelogEntry(
                version=self.version,
                action="remove_records",
                details=f"Removed {len(ids)} records",
                record_ids=list(ids),
            )
        )

    def retire_mastered(self, results: "EvalResults", streak: int = 5) -> list[str]:
        """Retire records that scored perfectly for `streak` consecutive runs.

        Since a single EvalResults only represents one run, this checks the
        current run's perfect scorers and is meant to be called repeatedly
        as part of a tracked history; callers wanting true multi-run streak
        tracking should track record ids externally and pass already
        streak-qualified ids via remove_records instead.
        """
        perfect_ids = [r.record_id for r in results.records if r.overall_score >= 1.0]
        if streak <= 1:
            self.remove_records(perfect_ids)
        return perfect_ids


# ---------------------------------------------------------------------------
# RecordResult / EvalResults
# ---------------------------------------------------------------------------


@dataclass
class RecordResult:
    record_id: str
    question: str
    question_type: QuestionType
    noise_profile: list[NoiseType]
    difficulty: Difficulty

    rag_answer: str
    rag_contexts: list[str]
    rag_error: str | None

    chunk_recall: float
    set_recall: bool
    mrr: float | None
    precision_at_k: float | None
    gold_chunk_ranks: list[int] | None

    faithfulness: DimensionScore
    correctness: DimensionScore
    abstention: DimensionScore
    completeness: DimensionScore | None
    conciseness: DimensionScore | None
    overall_score: float

    intent_coverage: float
    intent_details: list[IntentVerdict]

    discovered_alt_chunks: list[str] | None = None
    domain_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "question": self.question,
            "question_type": self.question_type.value,
            "noise_profile": [n.value for n in self.noise_profile],
            "difficulty": self.difficulty.value,
            "rag_answer": self.rag_answer,
            "rag_contexts": self.rag_contexts,
            "rag_error": self.rag_error,
            "chunk_recall": self.chunk_recall,
            "set_recall": self.set_recall,
            "mrr": self.mrr,
            "precision_at_k": self.precision_at_k,
            "gold_chunk_ranks": self.gold_chunk_ranks,
            "faithfulness": self.faithfulness.to_dict(),
            "correctness": self.correctness.to_dict(),
            "abstention": self.abstention.to_dict(),
            "completeness": self.completeness.to_dict() if self.completeness else None,
            "conciseness": self.conciseness.to_dict() if self.conciseness else None,
            "overall_score": self.overall_score,
            "intent_coverage": self.intent_coverage,
            "intent_details": [d.to_dict() for d in self.intent_details],
            "discovered_alt_chunks": self.discovered_alt_chunks,
            "domain_tags": self.domain_tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordResult":
        return cls(
            record_id=data["record_id"],
            question=data["question"],
            question_type=QuestionType(data["question_type"]),
            noise_profile=[NoiseType(n) for n in data.get("noise_profile", [])],
            difficulty=Difficulty(data["difficulty"]),
            rag_answer=data["rag_answer"],
            rag_contexts=data.get("rag_contexts", []),
            rag_error=data.get("rag_error"),
            chunk_recall=data["chunk_recall"],
            set_recall=data["set_recall"],
            mrr=data.get("mrr"),
            precision_at_k=data.get("precision_at_k"),
            gold_chunk_ranks=data.get("gold_chunk_ranks"),
            faithfulness=DimensionScore.from_dict(data["faithfulness"]),
            correctness=DimensionScore.from_dict(data["correctness"]),
            abstention=DimensionScore.from_dict(data["abstention"]),
            completeness=DimensionScore.from_dict(data["completeness"]) if data.get("completeness") else None,
            conciseness=DimensionScore.from_dict(data["conciseness"]) if data.get("conciseness") else None,
            overall_score=data["overall_score"],
            intent_coverage=data.get("intent_coverage", 0.0),
            intent_details=[IntentVerdict.from_dict(d) for d in data.get("intent_details", [])],
            discovered_alt_chunks=data.get("discovered_alt_chunks"),
            domain_tags=data.get("domain_tags", []),
        )


@dataclass
class AggregateScore:
    count: int
    mean_overall: float
    mean_faithfulness: float
    mean_correctness: float
    mean_abstention: float
    mean_completeness: float | None = None
    mean_conciseness: float | None = None
    mean_set_recall: float | None = None
    mean_chunk_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailurePatternReport:
    top_patterns: list[dict[str, Any]]
    worst_binary_questions: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {"top_patterns": self.top_patterns, "worst_binary_questions": self.worst_binary_questions}


@dataclass
class RetrievalReport:
    mean_set_recall: float
    mean_chunk_recall: float
    mean_mrr: float | None
    mean_precision_at_k: float | None
    by_question_type: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoringHealthReport:
    yes_rate_per_question: dict[str, float]
    yes_rate_spread_per_dimension: dict[str, float]
    phi_matrix: dict[str, dict[str, float]]
    high_correlation_pairs: list[tuple[str, str, float]]
    uncovered_failure_modes: list[str]
    verdict: str
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonReport:
    overall_delta: float
    dimension_deltas: dict[str, float]
    retrieval_deltas: dict[str, float]
    by_question_type_deltas: dict[str, float]
    by_noise_profile_deltas: dict[str, float]
    flipped_pass_to_fail: list[str]
    flipped_fail_to_pass: list[str]

    def summary(self) -> None:
        from litmus.report.comparison import print_comparison_summary

        print_comparison_summary(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResults:
    records: list[RecordResult]
    tier: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "metadata": self.metadata,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResults":
        return cls(
            tier=data["tier"],
            metadata=data.get("metadata", {}),
            records=[RecordResult.from_dict(r) for r in data.get("records", [])],
        )

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "EvalResults":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_html(self, path: str) -> None:
        from litmus.report.html_report import write_html_report

        write_html_report(self, path)

    def to_dataframe(self):
        from litmus.report.dataframe import results_to_dataframe

        return results_to_dataframe(self)

    # -- aggregates ---------------------------------------------------------

    def summary(self) -> None:
        from litmus.report.summary import print_results_summary

        print_results_summary(self)

    def _aggregate(self, records: list[RecordResult]) -> AggregateScore:
        n = len(records)
        if n == 0:
            return AggregateScore(0, 0.0, 0.0, 0.0, 0.0)
        has_completeness = any(r.completeness for r in records)
        has_conciseness = any(r.conciseness for r in records)
        has_mrr = any(r.mrr is not None for r in records)
        has_precision = any(r.precision_at_k is not None for r in records)
        return AggregateScore(
            count=n,
            mean_overall=sum(r.overall_score for r in records) / n,
            mean_faithfulness=sum(r.faithfulness.score for r in records) / n,
            mean_correctness=sum(r.correctness.score for r in records) / n,
            mean_abstention=sum(r.abstention.score for r in records) / n,
            mean_completeness=(sum(r.completeness.score for r in records if r.completeness) / n)
            if has_completeness
            else None,
            mean_conciseness=(sum(r.conciseness.score for r in records if r.conciseness) / n)
            if has_conciseness
            else None,
            mean_set_recall=sum(1.0 if r.set_recall else 0.0 for r in records) / n,
            mean_chunk_recall=sum(r.chunk_recall for r in records) / n,
        )

    def by_question_type(self) -> dict[QuestionType, AggregateScore]:
        out: dict[QuestionType, list[RecordResult]] = {}
        for r in self.records:
            out.setdefault(r.question_type, []).append(r)
        return {k: self._aggregate(v) for k, v in out.items()}

    def by_noise_profile(self) -> dict[str, AggregateScore]:
        out: dict[str, list[RecordResult]] = {}
        for r in self.records:
            key = "+".join(sorted(n.value for n in r.noise_profile)) or "clean"
            out.setdefault(key, []).append(r)
        return {k: self._aggregate(v) for k, v in out.items()}

    def by_difficulty(self) -> dict[Difficulty, AggregateScore]:
        out: dict[Difficulty, list[RecordResult]] = {}
        for r in self.records:
            out.setdefault(r.difficulty, []).append(r)
        return {k: self._aggregate(v) for k, v in out.items()}

    def by_domain(self) -> dict[str, AggregateScore]:
        out: dict[str, list[RecordResult]] = {}
        for r in self.records:
            for tag in r.domain_tags:
                out.setdefault(tag, []).append(r)
        return {k: self._aggregate(v) for k, v in out.items()}

    def failed_records(self, threshold: float = 0.7) -> list[RecordResult]:
        return [r for r in self.records if r.overall_score < threshold]

    def worst_dimensions(self) -> list[tuple[str, float]]:
        dims = ["faithfulness", "correctness", "abstention", "completeness", "conciseness"]
        scores: dict[str, list[float]] = {d: [] for d in dims}
        for r in self.records:
            scores["faithfulness"].append(r.faithfulness.score)
            scores["correctness"].append(r.correctness.score)
            scores["abstention"].append(r.abstention.score)
            if r.completeness:
                scores["completeness"].append(r.completeness.score)
            if r.conciseness:
                scores["conciseness"].append(r.conciseness.score)
        means = [(d, sum(v) / len(v)) for d, v in scores.items() if v]
        return sorted(means, key=lambda kv: kv[1])

    # -- diagnostics -----------------------------------------------------

    def failure_patterns(self) -> FailurePatternReport:
        from litmus.evaluate.diagnostics import compute_failure_patterns

        return compute_failure_patterns(self)

    def retrieval_summary(self) -> RetrievalReport:
        from litmus.evaluate.diagnostics import compute_retrieval_summary

        return compute_retrieval_summary(self)

    def scoring_health(self) -> ScoringHealthReport:
        from litmus.evaluate.diagnostics import compute_scoring_health

        return compute_scoring_health(self)

    # -- comparison -------------------------------------------------------

    def compare(self, other: "EvalResults") -> ComparisonReport:
        from litmus.report.comparison import compare_results

        return compare_results(self, other)


# ---------------------------------------------------------------------------
# Misc reporting models
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    generation_input_tokens: int
    generation_output_tokens: int
    generation_cost_usd: float
    evaluation_input_tokens: int
    evaluation_output_tokens: int
    evaluation_cost_usd: float
    total_cost_usd: float
    estimated_time_minutes: float
    question_count: int

    def __str__(self) -> str:
        return (
            f"Cost Estimate:\n"
            f"  Questions to generate: {self.question_count}\n"
            f"  Generation: ~{self.generation_input_tokens} in / {self.generation_output_tokens} out tokens, "
            f"${self.generation_cost_usd:.2f}\n"
            f"  Evaluation (1 run): ~{self.evaluation_input_tokens} in / {self.evaluation_output_tokens} out tokens, "
            f"${self.evaluation_cost_usd:.2f}\n"
            f"  Total: ${self.total_cost_usd:.2f}\n"
            f"  Estimated time: ~{self.estimated_time_minutes:.0f} minutes"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlaggedRecord:
    record_id: str
    failed_checks: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageReport:
    question_type_coverage: dict[str, bool]
    unanswerable_ratio: float
    domain_coverage_ok: bool
    noise_distribution_ok: bool
    difficulty_distribution_ok: bool
    cross_doc_ratio: float
    contradiction_ratio: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    overall_pass: bool
    flagged_records: list[FlaggedRecord]
    dimension_scores: dict[str, float]
    coverage_report: CoverageReport | None
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "flagged_records": [f.to_dict() for f in self.flagged_records],
            "dimension_scores": self.dimension_scores,
            "coverage_report": self.coverage_report.to_dict() if self.coverage_report else None,
            "recommendations": self.recommendations,
        }


@dataclass
class StaleRecord:
    record_id: str
    source_doc_ids: list[str]
    change_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StalenessReport:
    stale_records: list[StaleRecord]
    total_records: int
    stale_count: int
    stale_ratio: float
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stale_records": [r.to_dict() for r in self.stale_records],
            "total_records": self.total_records,
            "stale_count": self.stale_count,
            "stale_ratio": self.stale_ratio,
            "recommendations": self.recommendations,
        }
