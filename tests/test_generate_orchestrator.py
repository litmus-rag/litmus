from litmus.generate.orchestrator import _checkpoint_scope


def test_checkpoint_scope_differs_across_docs_dirs():
    a = _checkpoint_scope("/corpus/a", "minimal")
    b = _checkpoint_scope("/corpus/b", "minimal")
    assert a != b


def test_checkpoint_scope_differs_across_tiers():
    a = _checkpoint_scope("/corpus/a", "minimal")
    b = _checkpoint_scope("/corpus/a", "medium")
    assert a != b


def test_checkpoint_scope_stable_for_same_inputs():
    a1 = _checkpoint_scope("/corpus/a", "minimal")
    a2 = _checkpoint_scope("/corpus/a", "minimal")
    assert a1 == a2


def test_checkpoint_scope_resolves_relative_paths():
    # Two ways of referring to the same directory must scope identically,
    # or a relative-vs-absolute docs_dir call would defeat the scoping.
    a = _checkpoint_scope("/corpus/a", "minimal")
    b = _checkpoint_scope("/corpus/./a", "minimal")
    assert a == b
