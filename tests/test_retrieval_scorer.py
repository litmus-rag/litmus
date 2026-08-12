from litmus.evaluate.retrieval_scorer import chunk_is_retrieved, first_retrieval_rank, score_retrieval


def test_chunk_is_retrieved_exact_match():
    assert chunk_is_retrieved("The sky is blue.", ["Some prefix. The sky is blue. Some suffix."])


def test_chunk_is_retrieved_no_match():
    assert not chunk_is_retrieved("The sky is blue.", ["Completely unrelated sentence about cars."])


def test_chunk_is_retrieved_near_duplicate_via_jaccard():
    gold = "The Enterprise plan guarantees 99.9% uptime and 1-hour response times."
    retrieved = ["Enterprise plan guarantees a 99.9% uptime with 1 hour response time commitments."]
    assert chunk_is_retrieved(gold, retrieved, jaccard_threshold=0.5)


def test_chunk_is_retrieved_empty_gold_text():
    assert not chunk_is_retrieved("", ["anything"])


def test_first_retrieval_rank():
    gold = "The sky is blue."
    retrieved = ["unrelated one", "unrelated two", "The sky is blue today.", "unrelated three"]
    assert first_retrieval_rank(gold, retrieved) == 3


def test_first_retrieval_rank_none_when_absent():
    assert first_retrieval_rank("The sky is blue.", ["unrelated"]) is None


def test_score_retrieval_unanswerable_is_vacuously_perfect():
    result = score_retrieval([], ["some context"])
    assert result.chunk_recall == 1.0
    assert result.set_recall is True
    assert result.mrr is None


def test_score_retrieval_set_recall_requires_full_group():
    gold = [["chunk A text here.", "chunk B text here."]]
    # Only chunk A retrieved: set recall should fail even though chunk recall is partial.
    result = score_retrieval(gold, ["chunk A text here and more."])
    assert result.set_recall is False
    assert 0 < result.chunk_recall < 1


def test_score_retrieval_alternative_set_satisfies_recall():
    gold = [["chunk A text here.", "chunk B text here."], ["chunk C standalone text."]]
    result = score_retrieval(gold, ["chunk C standalone text and more detail."])
    assert result.set_recall is True


def test_score_retrieval_mrr_and_precision():
    gold = [["target chunk text"]]
    retrieved = ["distractor one", "target chunk text right here", "distractor two"]
    result = score_retrieval(gold, retrieved, compute_mrr=True, compute_precision=True, compute_ranks=True)
    assert result.mrr == 0.5  # rank 2
    assert result.precision_at_k == 1 / 3
    assert result.gold_chunk_ranks == [2]


def test_score_retrieval_no_retrieval_at_all():
    gold = [["target chunk text"]]
    result = score_retrieval(gold, [], compute_mrr=True, compute_precision=True)
    assert result.chunk_recall == 0.0
    assert result.set_recall is False
    assert result.mrr == 0.0
