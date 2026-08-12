"""Hypothesis-based fuzz tests for pure-logic modules: chunker, typo
injection, sizing formula, phi-coefficient edge cases. No LLM calls."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from litmus.evaluate.diagnostics import _phi_coefficient
from litmus.generate.noise import inject_typos
from litmus.generate.sizing import calculate_size
from litmus.ingest.chunker import chunk_text_auto, chunk_text_paragraphs, chunk_text_sentences
from litmus.llm.cost import count_tokens

# --- Chunker fuzzing ---------------------------------------------------


@given(st.text(min_size=0, max_size=2000))
@settings(max_examples=200, deadline=None)
def test_fuzz_chunk_text_auto_never_crashes(text):
    chunks = chunk_text_auto(text, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


@given(st.text(min_size=0, max_size=2000))
@settings(max_examples=200, deadline=None)
def test_fuzz_chunk_text_sentences_never_crashes(text):
    chunks = chunk_text_sentences(text, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunks, list)


@given(st.text(min_size=0, max_size=2000))
@settings(max_examples=200, deadline=None)
def test_fuzz_chunk_text_paragraphs_never_crashes(text):
    chunks = chunk_text_paragraphs(text, chunk_size=50, chunk_overlap=10)
    assert isinstance(chunks, list)


@given(
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=3000),
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=0, max_value=100),
)
@settings(max_examples=150, deadline=None)
def test_fuzz_chunker_preserves_all_nonwhitespace_content(text, chunk_size, chunk_overlap):
    """No chunking strategy should silently drop non-whitespace characters
    (overlap can duplicate content, but never lose it)."""
    chunks = chunk_text_auto(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    original_chars = sorted(c for c in text if not c.isspace())
    chunk_chars = sorted(c for c in "".join(chunks) if not c.isspace())
    for ch in set(original_chars):
        assert chunk_chars.count(ch) >= original_chars.count(ch) or original_chars.count(ch) == 0


# --- Typo injection fuzzing ---------------------------------------------


@given(
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu"), min_codepoint=65, max_codepoint=122), min_size=0, max_size=500),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=200, deadline=None)
def test_fuzz_inject_typos_never_crashes(text, num_typos, seed):
    result = inject_typos(text, num_typos=num_typos, seed=seed)
    assert isinstance(result, str)


@given(st.text(min_size=0, max_size=500), st.integers(min_value=0, max_value=10000))
@settings(max_examples=200, deadline=None)
def test_fuzz_inject_typos_arbitrary_unicode_never_crashes(text, seed):
    result = inject_typos(text, num_typos=1, seed=seed)
    assert isinstance(result, str)


@given(
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=200),
    st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=150, deadline=None)
def test_fuzz_inject_typos_preserves_word_count(text, seed):
    words_before = text.split(" ")
    result = inject_typos(text, num_typos=1, seed=seed)
    words_after = result.split(" ")
    assert len(words_after) == len(words_before)


# --- Sizing formula fuzzing ---------------------------------------------


@given(
    st.integers(min_value=1, max_value=200000),
    st.integers(min_value=0, max_value=2_000_000),
    st.sampled_from(["minimal", "medium", "exhaustive"]),
)
@settings(max_examples=200, deadline=None)
def test_fuzz_calculate_size_always_positive_and_bounded(num_docs, num_chunks, tier):
    size = calculate_size(num_docs, num_chunks, tier)
    assert size > 0
    # Should never wildly exceed the largest bracket's upper bound.
    assert size <= 700


@given(st.integers(min_value=1, max_value=100000), st.integers(min_value=0, max_value=1_000_000))
@settings(max_examples=100, deadline=None)
def test_fuzz_calculate_size_consistent_across_tiers(num_docs, num_chunks):
    minimal = calculate_size(num_docs, num_chunks, "minimal")
    medium = calculate_size(num_docs, num_chunks, "medium")
    exhaustive = calculate_size(num_docs, num_chunks, "exhaustive")
    # Higher tiers should never produce a smaller eval set for the same corpus.
    assert minimal <= medium <= exhaustive


# --- Phi-coefficient fuzzing ---------------------------------------------


@given(
    st.lists(st.booleans(), min_size=2, max_size=100),
    st.lists(st.booleans(), min_size=2, max_size=100),
)
@settings(max_examples=300, deadline=None)
def test_fuzz_phi_coefficient_bounded(a, b):
    n = min(len(a), len(b))
    phi = _phi_coefficient(a[:n], b[:n])
    assert -1.0 - 1e-9 <= phi <= 1.0 + 1e-9
    assert not math.isnan(phi)


@given(st.lists(st.booleans(), min_size=1, max_size=50))
@settings(max_examples=100, deadline=None)
def test_fuzz_phi_coefficient_self_correlation_is_one_or_zero(series):
    phi = _phi_coefficient(series, series)
    # A column with zero variance (all same value) has undefined phi, which
    # this implementation defines as 0.0; anything with variance correlates
    # perfectly with itself.
    if len(set(series)) > 1:
        assert phi == 1.0
    else:
        assert phi == 0.0


@given(st.lists(st.booleans(), min_size=0, max_size=0))
def test_fuzz_phi_coefficient_empty_series(_):
    assert _phi_coefficient([], []) == 0.0


# --- Token counting fuzzing ---------------------------------------------


@given(st.text(min_size=0, max_size=5000))
@settings(max_examples=150, deadline=None)
def test_fuzz_count_tokens_never_crashes_and_nonnegative(text):
    tokens = count_tokens(text)
    assert tokens >= 0
    if text.strip():
        assert tokens > 0
