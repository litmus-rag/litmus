"""Chunking with stable IDs.

Strategies:
- "auto": token-windowed chunking (chunk_size/chunk_overlap in tokens),
  splitting on paragraph boundaries where possible so chunks don't cut mid-
  sentence unnecessarily.
- "sentences": one or more sentences per chunk, packed up to chunk_size
  tokens.
- "paragraphs": one paragraph per chunk (further split if a single
  paragraph exceeds chunk_size tokens).
- callable: user-supplied ``fn(text: str) -> list[str]``.

Chunk IDs are stable across re-chunking runs of the same document with the
same strategy/size/overlap: ``{doc_id}#chunk{index}``.
"""

from __future__ import annotations

import re
from typing import Callable

from litmus.ingest.loader import LoadedDocument
from litmus.llm.cost import count_tokens
from litmus.models import Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    parts = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pack_units(units: list[str], chunk_size: int, chunk_overlap: int, joiner: str) -> list[str]:
    """Greedily pack text units (sentences/paragraphs) into token windows."""
    if not units:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > chunk_size:
            chunks.append(joiner.join(current))
            if chunk_overlap > 0:
                overlap_units: list[str] = []
                overlap_tokens = 0
                for prev in reversed(current):
                    t = count_tokens(prev)
                    if overlap_tokens + t > chunk_overlap:
                        break
                    overlap_units.insert(0, prev)
                    overlap_tokens += t
                current = overlap_units
                current_tokens = overlap_tokens
            else:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        chunks.append(joiner.join(current))
    return chunks


def chunk_text_auto(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    units: list[str] = []
    for para in paragraphs:
        if count_tokens(para) > chunk_size:
            units.extend(split_sentences(para))
        else:
            units.append(para)
    return _pack_units(units, chunk_size, chunk_overlap, joiner="\n\n")


def chunk_text_sentences(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    sentences = split_sentences(text)
    return _pack_units(sentences, chunk_size, chunk_overlap, joiner=" ")


def chunk_text_paragraphs(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    paragraphs = split_paragraphs(text)
    out: list[str] = []
    for para in paragraphs:
        if count_tokens(para) <= chunk_size:
            out.append(para)
        else:
            out.extend(_pack_units(split_sentences(para), chunk_size, chunk_overlap, joiner=" "))
    return out


_STRATEGIES: dict[str, Callable[[str, int, int], list[str]]] = {
    "auto": chunk_text_auto,
    "sentences": chunk_text_sentences,
    "paragraphs": chunk_text_paragraphs,
}


def chunk_document(
    doc: LoadedDocument,
    chunking: str | Callable[[str], list[str]] = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    if callable(chunking):
        pieces = chunking(doc.text)
    else:
        strategy = _STRATEGIES.get(chunking)
        if strategy is None:
            raise ValueError(f"Unknown chunking strategy: {chunking!r}. Use auto/sentences/paragraphs or a callable.")
        pieces = strategy(doc.text, chunk_size, chunk_overlap)

    chunks = []
    for i, piece in enumerate(pieces):
        if not piece.strip():
            continue
        chunks.append(
            Chunk(
                id=f"{doc.doc_id}#chunk{i}",
                doc_id=doc.doc_id,
                text=piece,
                index=i,
                token_count=count_tokens(piece),
            )
        )
    return chunks


def chunk_documents(
    docs: list[LoadedDocument],
    chunking: str | Callable[[str], list[str]] = "auto",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict[str, Chunk]:
    all_chunks: dict[str, Chunk] = {}
    for doc in docs:
        for chunk in chunk_document(doc, chunking, chunk_size, chunk_overlap):
            all_chunks[chunk.id] = chunk
    return all_chunks
