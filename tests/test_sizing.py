import pytest

from litmus.generate.sizing import calculate_size, depth_for_document, document_coverage_target


@pytest.mark.parametrize("tier", ["minimal", "medium", "exhaustive"])
def test_calculate_size_within_bracket_range(tier):
    from litmus.config import SIZING_TABLE

    for upper, ranges in SIZING_TABLE:
        num_docs = upper if upper is not None else 6000
        low, high = ranges[tier]
        size = calculate_size(num_docs, num_docs * 10, tier)
        assert low <= size <= high


def test_calculate_size_zero_docs_raises():
    with pytest.raises(ValueError):
        calculate_size(0, 0, "minimal")


def test_calculate_size_unknown_tier_raises():
    with pytest.raises(ValueError):
        calculate_size(10, 100, "nonexistent")


def test_calculate_size_monotonic_in_density():
    sparse = calculate_size(100, 100, "medium")
    dense = calculate_size(100, 5000, "medium")
    assert dense >= sparse


def test_document_coverage_target_minimal_is_none():
    assert document_coverage_target("minimal") is None


def test_document_coverage_target_medium_and_exhaustive():
    medium = document_coverage_target("medium")
    exhaustive = document_coverage_target("exhaustive")
    assert medium == (0.15, 0.30)
    assert exhaustive == (0.20, 0.40)


@pytest.mark.parametrize(
    "num_chunks,expected",
    [(1, (1, 1)), (3, (1, 1)), (5, (2, 3)), (10, (2, 3)), (11, (3, 5)), (50, (3, 5))],
)
def test_depth_for_document(num_chunks, expected):
    assert depth_for_document(num_chunks) == expected
