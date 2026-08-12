from litmus.evaluate.orchestrator import _checkpoint_key
from litmus.models import EvalSetMetadata, EvalSet


def _make_eval_set(docs_dir="/corpus/a", generated_at="123", tier="minimal"):
    return EvalSet(
        records=[],
        chunks={},
        metadata=EvalSetMetadata(docs_dir=docs_dir, generated_at=generated_at),
        tier=tier,
    )


def test_checkpoint_key_differs_across_docs_dirs():
    a = _checkpoint_key(_make_eval_set(docs_dir="/corpus/a"), None)
    b = _checkpoint_key(_make_eval_set(docs_dir="/corpus/b"), None)
    assert a != b


def test_checkpoint_key_differs_across_generation_runs():
    # Two eval sets generated from the same docs_dir at different times
    # (generated_at differs) must not collide -- they may contain
    # different records even over the same corpus.
    a = _checkpoint_key(_make_eval_set(generated_at="111"), None)
    b = _checkpoint_key(_make_eval_set(generated_at="222"), None)
    assert a != b


def test_checkpoint_key_differs_across_scoring_configs():
    a = _checkpoint_key(_make_eval_set(), "mvp")
    b = _checkpoint_key(_make_eval_set(), "full")
    assert a != b


def test_checkpoint_key_stable_for_same_inputs():
    a1 = _checkpoint_key(_make_eval_set(), "mvp")
    a2 = _checkpoint_key(_make_eval_set(), "mvp")
    assert a1 == a2
