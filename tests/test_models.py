import json
import tempfile
from pathlib import Path

from litmus.models import (
    BinaryQuestion,
    BinaryVerdict,
    Chunk,
    Difficulty,
    DimensionScore,
    EvalRecord,
    EvalSet,
    EvalSetMetadata,
    IntentPoint,
    NoiseType,
    QuestionType,
    RAGResponse,
    normalize_rag_response,
)


def _sample_record(record_id="eval-0001"):
    return EvalRecord(
        id=record_id,
        question="How much PTO carries over?",
        question_clean="How much PTO carries over?",
        question_type=QuestionType.CONTRADICTION,
        noise_profile=[NoiseType.CLEAN],
        difficulty=Difficulty.HARD,
        gold_answer="Sources disagree; the newer policy caps it at 40 hours.",
        gold_chunk_ids=[["hr_pto_2023#chunk0", "hr_benefits_2025#chunk0"]],
        gold_chunks_text=[["old text", "new text"]],
        intent_points=[IntentPoint(id="P1", text="acknowledges conflict", required=True)],
        domain_tags=["hr_pto_2023", "hr_benefits_2025"],
        source_doc_ids=["hr_pto_2023", "hr_benefits_2025"],
    )


def test_eval_record_roundtrip_dict():
    record = _sample_record()
    data = record.to_dict()
    restored = EvalRecord.from_dict(data)
    assert restored == record


def test_eval_set_save_and_load_roundtrip():
    record = _sample_record()
    chunk = Chunk(id="hr_pto_2023#chunk0", doc_id="hr_pto_2023", text="old text", index=0, token_count=3)
    eval_set = EvalSet(
        records=[record],
        chunks={chunk.id: chunk},
        metadata=EvalSetMetadata(docs_dir="./docs", llm="azure/gpt-5.4"),
        tier="medium",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "eval_set.json")
        eval_set.save(path)
        loaded = EvalSet.load(path)
    assert len(loaded.records) == 1
    assert loaded.records[0].question == record.question
    assert loaded.tier == "medium"
    assert loaded.chunks["hr_pto_2023#chunk0"].text == "old text"


def test_eval_set_filter_by_question_type():
    r1 = _sample_record("r1")
    r2 = _sample_record("r2")
    r2.question_type = QuestionType.SINGLE_CHUNK
    eval_set = EvalSet(records=[r1, r2], chunks={}, metadata=EvalSetMetadata(), tier="medium")
    filtered = eval_set.filter(question_type=QuestionType.SINGLE_CHUNK)
    assert len(filtered.records) == 1
    assert filtered.records[0].id == "r2"


def test_eval_set_add_and_remove_records_bumps_version():
    eval_set = EvalSet(records=[], chunks={}, metadata=EvalSetMetadata(), tier="medium", version="1.0.0")
    eval_set.add_records([_sample_record("r1")])
    assert eval_set.version == "1.1.0"
    assert len(eval_set.records) == 1
    eval_set.remove_records(["r1"])
    assert eval_set.version == "1.2.0"
    assert len(eval_set.records) == 0


def test_eval_set_document_coverage():
    r1 = _sample_record("r1")
    chunks = {
        "hr_pto_2023#chunk0": Chunk(id="hr_pto_2023#chunk0", doc_id="hr_pto_2023", text="t", index=0),
        "hr_benefits_2025#chunk0": Chunk(id="hr_benefits_2025#chunk0", doc_id="hr_benefits_2025", text="t", index=0),
        "other_doc#chunk0": Chunk(id="other_doc#chunk0", doc_id="other_doc", text="t", index=0),
    }
    eval_set = EvalSet(records=[r1], chunks=chunks, metadata=EvalSetMetadata(), tier="medium")
    assert eval_set.document_coverage == 2 / 3


def test_normalize_rag_response_dict():
    resp = normalize_rag_response({"answer": "a", "contexts": ["c1", "c2"]})
    assert resp == RAGResponse(answer="a", contexts=["c1", "c2"])


def test_normalize_rag_response_tuple():
    resp = normalize_rag_response(("a", ["c1"]))
    assert resp == RAGResponse(answer="a", contexts=["c1"])


def test_normalize_rag_response_already_rag_response():
    original = RAGResponse(answer="a", contexts=[])
    assert normalize_rag_response(original) is original


def test_normalize_rag_response_invalid_type_raises():
    try:
        normalize_rag_response(42)
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_dimension_score_roundtrip():
    score = DimensionScore(score=0.75, verdicts={"F1": BinaryVerdict(question_id="F1", verdict=True, reason="ok")})
    data = score.to_dict()
    restored = DimensionScore.from_dict(data)
    assert restored.score == 0.75
    assert restored.verdicts["F1"].verdict is True


def test_binary_question_defaults():
    q = BinaryQuestion(id="X1", dimension="custom", text="Is it good?")
    assert q.weight == 1.0
