"""Live end-to-end tests against the real gpt-5.4 deployment and the actual
sample1/ drug-label PDF corpus. No mocking anywhere in this file - if
credentials in `conf` aren't available, tests skip rather than falling back
to a mock (per project policy: verify against the real system or skip).
"""

from pathlib import Path

import pytest

from litmus.llm.client import LLMClient, credentials_available

pytestmark = pytest.mark.live

SAMPLE1_DIR = str(Path(__file__).parent.parent / "sample1")


def _skip_if_no_llm():
    if not credentials_available():
        pytest.skip("Azure credentials not available (conf file missing or incomplete)")


@pytest.fixture(scope="module", autouse=True)
def _require_llm():
    _skip_if_no_llm()


@pytest.fixture(scope="module")
def sample1_chunks():
    from litmus.ingest.chunker import chunk_documents
    from litmus.ingest.loader import load_directory

    docs = load_directory(SAMPLE1_DIR)
    return chunk_documents(docs, chunk_size=400, chunk_overlap=50)


def test_load_and_chunk_sample1_corpus(sample1_chunks):
    assert len(sample1_chunks) > 100
    doc_ids = {c.doc_id for c in sample1_chunks.values()}
    assert "ozempic" in doc_ids
    assert "mounjaro" in doc_ids


def test_llm_client_basic_completion():
    client = LLMClient()
    reply = client.complete("Reply with exactly the word: pong", max_tokens=10)
    assert "pong" in reply.lower()


def test_llm_client_json_completion():
    client = LLMClient()
    result = client.complete_json('Return this exact JSON object and nothing else: {"status": "ok"}', max_tokens=50)
    assert result == {"status": "ok"}


def test_generate_single_chunk_question_from_real_drug_label(sample1_chunks):
    from litmus.generate.question_types import generate_single_chunk

    client = LLMClient()
    chunk = next(c for c in sample1_chunks.values() if "ozempic" in c.doc_id.lower() and len(c.text) > 200)
    result = generate_single_chunk(chunk, client)
    assert result["question"].strip()
    assert result["gold_answer"].strip()
    assert result["gold_chunk_ids"] == [[chunk.id]]


def test_generate_unanswerable_question(sample1_chunks):
    from litmus.generate.question_types import generate_unanswerable

    client = LLMClient()
    sample_chunks = list(sample1_chunks.values())[:8]
    result = generate_unanswerable(sample_chunks, client)
    assert result["unanswerable"] is True
    assert result["gold_chunk_ids"] == []


def test_leakage_filter_flags_common_knowledge():
    from litmus.generate.leakage import check_leakage

    client = LLMClient()
    # A question with an obviously well-known answer should trip the leakage check.
    leaked = check_leakage("What is the capital of France?", "The capital of France is Paris.", client)
    assert leaked is True


def test_leakage_filter_passes_obscure_specific_claim():
    from litmus.generate.leakage import check_leakage

    client = LLMClient()
    # A specific, made-up-sounding numeric claim the model can't know from memory.
    leaked = check_leakage(
        "What is the internal SKU code for the enterprise widget bundle?",
        "The internal SKU code for the enterprise widget bundle is WX-88213-ENT.",
        client,
    )
    assert leaked is False


def test_full_generate_pipeline_minimal_tier(live_run_dir):
    from litmus.generate.orchestrator import generate_eval_set

    save_path = live_run_dir / "test_full_generate_pipeline_minimal_tier" / "eval_set.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    eval_set = generate_eval_set(
        docs_dir=SAMPLE1_DIR,
        llm="azure/gpt-5.4",
        tier="minimal",
        size=4,
        chunk_size=400,
        chunk_overlap=50,
        save_path=str(save_path),
        cache_dir=str(live_run_dir / "test_full_generate_pipeline_minimal_tier" / "cache"),
        max_workers=4,
        seed=123,
        verbose=True,
    )
    assert 1 <= len(eval_set.records) <= 4
    for record in eval_set.records:
        assert record.question.strip()
        if not record.unanswerable:
            assert record.gold_answer.strip()
            assert record.gold_chunk_ids
    assert save_path.exists()
    print(f"\nGenerated eval set saved to: {save_path.resolve()}")


def test_full_evaluate_pipeline_against_generated_set(live_run_dir):
    from litmus.evaluate.orchestrator import evaluate_eval_set
    from litmus.generate.orchestrator import generate_eval_set

    run_dir = live_run_dir / "test_full_evaluate_pipeline_against_generated_set"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_set_path = run_dir / "eval_set.json"
    results_path = run_dir / "results.json"

    eval_set = generate_eval_set(
        docs_dir=SAMPLE1_DIR,
        llm="azure/gpt-5.4",
        tier="minimal",
        size=3,
        chunk_size=400,
        chunk_overlap=50,
        save_path=str(eval_set_path),
        cache_dir=str(run_dir / "gen_cache"),
        max_workers=4,
        seed=99,
        verbose=True,
    )
    assert len(eval_set.records) >= 1

    def stub_rag(question: str):
        # A deliberately mediocre RAG: returns the first available chunk
        # regardless of the question, to exercise both pass and fail paths.
        first_chunk = next(iter(eval_set.chunks.values()))
        return {"answer": f"According to the label: {first_chunk.text[:300]}", "contexts": [first_chunk.text]}

    results = evaluate_eval_set(
        eval_set,
        rag=stub_rag,
        max_workers=4,
        save_path=str(results_path),
        cache_dir=str(run_dir / "eval_cache"),
        verbose=True,
    )
    assert len(results.records) == len(eval_set.records)
    for record_result in results.records:
        assert 0.0 <= record_result.overall_score <= 1.0
        assert 0.0 <= record_result.faithfulness.score <= 1.0
    assert results_path.exists()
    print(f"\nEval set saved to:  {eval_set_path.resolve()}")
    print(f"Results saved to:  {results_path.resolve()}")


def test_evaluate_with_perfect_rag_scores_high(live_run_dir):
    """Sanity check: a RAG that returns the gold answer verbatim with the
    exact gold chunks should score near-perfect, proving the judge isn't
    systematically biased downward."""
    from litmus.evaluate.orchestrator import evaluate_eval_set
    from litmus.generate.orchestrator import generate_eval_set

    run_dir = live_run_dir / "test_evaluate_with_perfect_rag_scores_high"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_set = generate_eval_set(
        docs_dir=SAMPLE1_DIR,
        llm="azure/gpt-5.4",
        tier="minimal",
        size=2,
        chunk_size=400,
        chunk_overlap=50,
        save_path=str(run_dir / "eval_set.json"),
        cache_dir=str(run_dir / "gen_cache"),
        max_workers=2,
        seed=55,
        verbose=True,
    )
    answerable = [r for r in eval_set.records if not r.unanswerable]
    if not answerable:
        pytest.skip("No answerable records generated in this run")

    def perfect_rag(question: str):
        record = next(r for r in eval_set.records if r.question == question)
        contexts = [text for group in record.gold_chunks_text for text in group]
        return {"answer": record.gold_answer, "contexts": contexts}

    results_path = run_dir / "results.json"
    results = evaluate_eval_set(
        eval_set,
        rag=perfect_rag,
        max_workers=2,
        save_path=str(results_path),
        cache_dir=str(run_dir / "eval_cache"),
        verbose=True,
    )
    mean_overall = sum(r.overall_score for r in results.records) / len(results.records)
    assert mean_overall >= 0.7
    print(f"\nResults saved to: {results_path.resolve()}")


def test_contradiction_detection_on_hr_fixture_corpus(fixtures_corpus_dir):
    from litmus.ingest.chunker import chunk_documents
    from litmus.ingest.contradiction_detector import find_contradictions
    from litmus.ingest.loader import load_directory

    docs = load_directory(fixtures_corpus_dir)
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)
    client = LLMClient()
    pairs = find_contradictions(chunks, client, similarity_threshold=0.5)
    # The fixture corpus deliberately contains a PTO-carryover contradiction
    # between hr_pto_2023.md and hr_benefits_2025.md.
    assert any(
        {chunks[p.chunk_id_a].doc_id, chunks[p.chunk_id_b].doc_id} == {"hr_pto_2023", "hr_benefits_2025"}
        for p in pairs
    )


def test_answer_scorer_catches_fabricated_fact():
    from litmus.evaluate.answer_scorer import MVP_QUESTIONS, aggregate_dimension_scores, run_judge

    client = LLMClient()
    question = "What is the maximum file upload size on the Pro plan?"
    context = ["The Pro plan raises the file upload limit to 100 MB per file."]
    bad_answer = "The Pro plan allows file uploads up to 750 MB per file."
    gold = "The maximum file upload size on the Pro plan is 100 MB."

    verdicts = run_judge(question, context, bad_answer, gold, False, MVP_QUESTIONS, client)
    scores = aggregate_dimension_scores(verdicts, MVP_QUESTIONS)
    assert scores["faithfulness"].score < 1.0


def test_answer_scorer_rewards_correct_faithful_answer():
    from litmus.evaluate.answer_scorer import MVP_QUESTIONS, aggregate_dimension_scores, run_judge

    client = LLMClient()
    question = "What is the maximum file upload size on the Pro plan?"
    context = ["The Pro plan raises the file upload limit to 100 MB per file."]
    good_answer = "The Pro plan allows file uploads up to 100 MB per file."
    gold = "The maximum file upload size on the Pro plan is 100 MB."

    verdicts = run_judge(question, context, good_answer, gold, False, MVP_QUESTIONS, client)
    scores = aggregate_dimension_scores(verdicts, MVP_QUESTIONS)
    assert scores["faithfulness"].score == 1.0
    assert scores["correctness"].score == 1.0
