from litmus.evaluate.diagnostics import (
    _phi_coefficient,
    compute_phi_matrix,
    compute_scoring_health,
    compute_yes_rate_spread_per_dimension,
    compute_yes_rates,
    detect_stub_rag,
    high_correlation_pairs,
)
from litmus.models import (
    BinaryVerdict,
    Difficulty,
    DimensionScore,
    EvalResults,
    NoiseType,
    QuestionType,
    RecordResult,
)


def _make_result(record_id, f_verdicts, c_verdicts, a_verdicts, overall=1.0, answer="a", contexts=None):
    def dim(verdicts_dict):
        return DimensionScore(
            score=sum(1 for v in verdicts_dict.values() if v) / len(verdicts_dict),
            verdicts={qid: BinaryVerdict(question_id=qid, verdict=v) for qid, v in verdicts_dict.items()},
        )

    return RecordResult(
        record_id=record_id,
        question="q",
        question_type=QuestionType.SINGLE_CHUNK,
        noise_profile=[NoiseType.CLEAN],
        difficulty=Difficulty.EASY,
        rag_answer=answer,
        rag_contexts=contexts or [],
        rag_error=None,
        chunk_recall=1.0,
        set_recall=True,
        mrr=None,
        precision_at_k=None,
        gold_chunk_ranks=None,
        faithfulness=dim(f_verdicts),
        correctness=dim(c_verdicts),
        abstention=dim(a_verdicts),
        completeness=None,
        conciseness=None,
        overall_score=overall,
        intent_coverage=1.0,
        intent_details=[],
    )


def test_phi_coefficient_perfect_correlation():
    a = [True, True, False, False, True, False]
    b = [True, True, False, False, True, False]
    assert _phi_coefficient(a, b) == 1.0


def test_phi_coefficient_perfect_anticorrelation():
    a = [True, True, False, False]
    b = [False, False, True, True]
    assert _phi_coefficient(a, b) == -1.0


def test_phi_coefficient_no_variance_is_zero():
    # One question always yes: denominator is zero, phi should be defined as 0.
    a = [True, True, True, True]
    b = [True, False, True, False]
    assert _phi_coefficient(a, b) == 0.0


def test_phi_coefficient_independent_columns_near_zero():
    a = [True, False, True, False, True, False, True, False]
    b = [True, True, False, False, True, True, False, False]
    phi = _phi_coefficient(a, b)
    assert abs(phi) < 1.0


def test_compute_yes_rates_basic():
    results = EvalResults(
        records=[
            _make_result("r1", {"F1": True, "F2": True}, {"C1": True}, {"A1": False}),
            _make_result("r2", {"F1": False, "F2": True}, {"C1": True}, {"A1": True}),
        ],
        tier="minimal",
    )
    rates = compute_yes_rates(results)
    assert rates["F1"] == 0.5
    assert rates["F2"] == 1.0
    assert rates["C1"] == 1.0
    assert rates["A1"] == 0.5


def test_compute_yes_rate_spread_per_dimension():
    results = EvalResults(
        records=[
            _make_result("r1", {"F1": True, "F2": True}, {"C1": True}, {"A1": True}),
            _make_result("r2", {"F1": False, "F2": True}, {"C1": True}, {"A1": True}),
        ],
        tier="minimal",
    )
    spread = compute_yes_rate_spread_per_dimension(results)
    assert spread["faithfulness"] == 0.5  # F1=0.5, F2=1.0 -> spread 0.5
    assert spread["correctness"] == 0.0


def test_compute_phi_matrix_diagonal_is_one():
    results = EvalResults(
        records=[
            _make_result("r1", {"F1": True, "F2": False}, {"C1": True}, {"A1": True}),
            _make_result("r2", {"F1": False, "F2": True}, {"C1": True}, {"A1": True}),
        ],
        tier="minimal",
    )
    matrix = compute_phi_matrix(results)
    assert matrix["F1"]["F1"] == 1.0
    assert matrix["F2"]["F2"] == 1.0


def test_high_correlation_pairs_detects_near_duplicates():
    results = EvalResults(
        records=[
            _make_result("r1", {"F1": True, "F2": True}, {"C1": True}, {"A1": True}),
            _make_result("r2", {"F1": False, "F2": False}, {"C1": True}, {"A1": True}),
            _make_result("r3", {"F1": True, "F2": True}, {"C1": False}, {"A1": False}),
            _make_result("r4", {"F1": False, "F2": False}, {"C1": True}, {"A1": True}),
        ],
        tier="minimal",
    )
    matrix = compute_phi_matrix(results)
    pairs = high_correlation_pairs(matrix, threshold=0.5)
    pair_ids = {frozenset((a, b)) for a, b, _ in pairs}
    assert frozenset(("F1", "F2")) in pair_ids


def test_scoring_health_verdict_broken_on_narrow_spread():
    # All questions always pass -> spread near 0 -> broken/needs_work verdict.
    results = EvalResults(
        records=[_make_result(f"r{i}", {"F1": True, "F2": True}, {"C1": True}, {"A1": True}) for i in range(5)],
        tier="minimal",
    )
    health = compute_scoring_health(results)
    assert health.verdict in ("needs_work", "broken")


def test_scoring_health_reports_sane_fields():
    # Verdict category is sensitive to exact hand-crafted correlations, so
    # this asserts the report's structure/ranges are sane rather than
    # pinning a specific verdict string.
    results = EvalResults(
        records=[
            _make_result(
                "r1", {"F1": True, "F2": False, "F3": True}, {"C1": True, "C2": False}, {"A1": False, "A2": True}
            ),
            _make_result(
                "r2", {"F1": False, "F2": True, "F3": True}, {"C1": False, "C2": True}, {"A1": True, "A2": False}
            ),
            _make_result(
                "r3", {"F1": True, "F2": True, "F3": False}, {"C1": True, "C2": False}, {"A1": False, "A2": True}
            ),
            _make_result(
                "r4", {"F1": False, "F2": False, "F3": True}, {"C1": False, "C2": True}, {"A1": True, "A2": False}
            ),
            _make_result(
                "r5", {"F1": True, "F2": False, "F3": True}, {"C1": True, "C2": False}, {"A1": False, "A2": True}
            ),
        ],
        tier="minimal",
    )
    health = compute_scoring_health(results)
    assert health.verdict in ("healthy", "needs_work", "broken")
    for rate in health.yes_rate_per_question.values():
        assert 0.0 <= rate <= 1.0
    for spread in health.yes_rate_spread_per_dimension.values():
        assert 0.0 <= spread <= 1.0


def _f(v=True):
    return {"F1": v}


def _c(v=True):
    return {"C1": v}


def _a(v=True):
    return {"A1": v}


def test_detect_stub_rag_flags_quickstart_placeholder():
    results = [
        _make_result("r1", _f(), _c(), _a(), answer="...", contexts=["chunk text 1", "chunk text 2"]),
        _make_result("r2", _f(), _c(), _a(), answer="...", contexts=["chunk text 1", "chunk text 2"]),
    ]
    warnings = detect_stub_rag(results)
    assert warnings
    assert any("placeholder" in w.lower() for w in warnings)


def test_detect_stub_rag_flags_constant_response():
    results = [
        _make_result("r1", _f(), _c(), _a(), answer="the same answer every time", contexts=["ctx"]),
        _make_result("r2", _f(), _c(), _a(), answer="the same answer every time", contexts=["ctx"]),
        _make_result("r3", _f(), _c(), _a(), answer="the same answer every time", contexts=["ctx"]),
    ]
    warnings = detect_stub_rag(results)
    assert warnings
    assert any("same answer" in w.lower() for w in warnings)


def test_detect_stub_rag_no_warning_for_varied_real_looking_answers():
    results = [
        _make_result("r1", _f(), _c(), _a(), answer="The Pro plan allows 100 MB uploads.", contexts=["Pro plan chunk about uploads."]),
        _make_result("r2", _f(), _c(), _a(), answer="SSO is available on Enterprise via Okta.", contexts=["Enterprise SSO chunk."]),
        _make_result("r3", _f(), _c(), _a(), answer="PTO caps at 40 hours as of 2025.", contexts=["HR policy chunk."]),
    ]
    warnings = detect_stub_rag(results)
    assert warnings == []


def test_detect_stub_rag_empty_results_no_crash():
    assert detect_stub_rag([]) == []


def test_detect_stub_rag_single_record_skips_constant_check():
    # A single record can't demonstrate "same answer every time" - only the
    # placeholder-text check should apply.
    results = [_make_result("r1", _f(), _c(), _a(), answer="a real-sounding one-off answer", contexts=["real context"])]
    assert detect_stub_rag(results) == []
