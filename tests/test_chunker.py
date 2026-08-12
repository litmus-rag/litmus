from litmus.ingest.chunker import (
    chunk_text_auto,
    chunk_text_paragraphs,
    chunk_text_sentences,
    split_paragraphs,
    split_sentences,
)
from litmus.llm.cost import count_tokens
from litmus.models import Chunk
from litmus.ingest.loader import LoadedDocument
from litmus.ingest.chunker import chunk_document, chunk_documents


def test_split_sentences_basic():
    text = "This is one. This is two! Is this three?"
    sentences = split_sentences(text)
    assert sentences == ["This is one.", "This is two!", "Is this three?"]


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_paragraphs_basic():
    text = "Para one.\n\nPara two.\n\n\nPara three."
    paras = split_paragraphs(text)
    assert paras == ["Para one.", "Para two.", "Para three."]


def test_chunk_text_auto_respects_chunk_size():
    text = "\n\n".join(f"Paragraph number {i} with some extra words to pad it out a little." for i in range(20))
    chunks = chunk_text_auto(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c) <= 50 + 20  # small slack for overlap packing edge cases


def test_chunk_text_auto_empty_text():
    assert chunk_text_auto("") == []
    assert chunk_text_auto("   \n\n  ") == []


def test_chunk_text_sentences_packs_multiple_sentences():
    text = " ".join(f"Sentence {i} is short." for i in range(10))
    chunks = chunk_text_sentences(text, chunk_size=20, chunk_overlap=0)
    assert len(chunks) >= 2
    reconstructed_words = " ".join(chunks).split()
    original_words = text.split()
    # No words should be dropped (overlap=0 so no duplication either).
    assert reconstructed_words == original_words


def test_chunk_text_overlap_produces_repeated_content():
    text = " ".join(f"word{i}" for i in range(30))
    no_overlap = chunk_text_sentences(text, chunk_size=5, chunk_overlap=0)
    with_overlap = chunk_text_sentences(text, chunk_size=5, chunk_overlap=2)
    # Overlapping should never produce fewer total tokens across chunks than no-overlap.
    assert sum(count_tokens(c) for c in with_overlap) >= sum(count_tokens(c) for c in no_overlap)


def test_chunk_document_generates_stable_ids():
    doc = LoadedDocument(doc_id="mydoc", path="mydoc.md", text="Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks_a = chunk_document(doc, chunking="paragraphs")
    chunks_b = chunk_document(doc, chunking="paragraphs")
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert chunks_a[0].id == "mydoc#chunk0"
    assert all(c.doc_id == "mydoc" for c in chunks_a)


def test_chunk_document_callable_strategy():
    doc = LoadedDocument(doc_id="d", path="d.txt", text="a b c d e f")
    chunks = chunk_document(doc, chunking=lambda text: text.split())
    assert len(chunks) == 6
    assert chunks[0].text == "a"


def test_chunk_documents_merges_across_docs():
    docs = [
        LoadedDocument(doc_id="a", path="a.txt", text="Content of doc a."),
        LoadedDocument(doc_id="b", path="b.txt", text="Content of doc b."),
    ]
    chunks = chunk_documents(docs, chunking="paragraphs")
    assert any(cid.startswith("a#") for cid in chunks)
    assert any(cid.startswith("b#") for cid in chunks)


def test_chunk_document_unknown_strategy_raises():
    doc = LoadedDocument(doc_id="d", path="d.txt", text="text")
    try:
        chunk_document(doc, chunking="not_a_real_strategy")
        assert False, "expected ValueError"
    except ValueError:
        pass
