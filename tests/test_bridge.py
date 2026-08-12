from collections import Counter

from litmus.generate.bridge import _normalize_entity, _stratified_sample, pick_cross_doc_pair, BridgeCandidate
from litmus.models import Chunk


def _make_chunks(doc_chunk_counts: dict[str, int]) -> dict[str, Chunk]:
    chunks = {}
    for doc_id, n in doc_chunk_counts.items():
        for i in range(n):
            c = Chunk(id=f"{doc_id}#chunk{i}", doc_id=doc_id, text="x", index=i)
            chunks[c.id] = c
    return chunks


def test_stratified_sample_covers_all_docs_evenly():
    chunks = _make_chunks({f"doc{i}": 10 for i in range(5)})
    sample = _stratified_sample(chunks, 15)
    assert len(sample) == 15
    doc_counts = Counter(c.doc_id for c in sample)
    assert len(doc_counts) == 5
    assert all(count == 3 for count in doc_counts.values())


def test_stratified_sample_prefers_breadth_over_depth_for_uneven_docs():
    # One large doc, several small ones -- a naive prefix would be dominated
    # by the large doc; stratified sampling should still touch every doc.
    chunks = _make_chunks({"big": 100, "small1": 2, "small2": 2, "small3": 2})
    sample = _stratified_sample(chunks, 10)
    doc_counts = Counter(c.doc_id for c in sample)
    assert set(doc_counts) == {"big", "small1", "small2", "small3"}


def test_stratified_sample_limit_exceeds_total_chunks():
    chunks = _make_chunks({"doc0": 3, "doc1": 2})
    sample = _stratified_sample(chunks, 100)
    assert len(sample) == 5


def test_stratified_sample_empty_chunks():
    assert _stratified_sample({}, 10) == []


def test_normalize_entity_case_and_whitespace_insensitive():
    assert _normalize_entity("  Semaglutide  ") == "semaglutide"
    assert _normalize_entity("Semaglutide") == _normalize_entity("SEMAGLUTIDE")
    assert _normalize_entity("multi\n\tword   term") == "multi word term"


def test_pick_cross_doc_pair_returns_chunks_from_different_docs():
    chunks = _make_chunks({"docA": 2, "docB": 2})
    candidate = BridgeCandidate(
        entity="semaglutide",
        chunk_ids=["docA#chunk0", "docB#chunk0"],
        doc_ids=["docA", "docB"],
    )
    pair = pick_cross_doc_pair(candidate, chunks)
    assert pair is not None
    chunk_a, chunk_b = pair
    assert chunk_a.doc_id != chunk_b.doc_id


def test_pick_cross_doc_pair_returns_none_for_single_doc_candidate():
    chunks = _make_chunks({"docA": 2})
    candidate = BridgeCandidate(entity="x", chunk_ids=["docA#chunk0", "docA#chunk1"], doc_ids=["docA"])
    assert pick_cross_doc_pair(candidate, chunks) is None
