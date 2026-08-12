"""Local chunk embeddings via sentence-transformers (no external API needed).

Uses ``all-MiniLM-L6-v2`` — small, fast, CPU-friendly, good enough for the
similarity pre-filtering this library needs (contradiction candidate pairs,
conceptual "similar chunk" lookups). Not tied to the Azure resource, since
CLAUDE.md warns no embeddings deployment is confirmed live there.

The model is downloaded from the Hugging Face Hub on first use. On this
machine, Python's default SSL context fails to verify huggingface.co
(certifi bundle issue) even though the system trust store is fine, so we
inject ``truststore`` before importing anything network-touching.
"""

from __future__ import annotations

import truststore

truststore.inject_into_ssl()

import numpy as np  # noqa: E402

_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalized embeddings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Assumes rows are already L2-normalized."""
    return embeddings @ embeddings.T


def top_similar_pairs(
    embeddings: np.ndarray,
    ids: list[str],
    threshold: float = 0.8,
    exclude_same_doc: dict[str, str] | None = None,
) -> list[tuple[str, str, float]]:
    """Return (id_a, id_b, similarity) for all pairs above `threshold`.

    If `exclude_same_doc` (chunk_id -> doc_id) is given, pairs from the same
    document are skipped — contradiction detection only cares about
    cross-document conflicts.
    """
    if len(ids) < 2:
        return []
    sims = cosine_similarity_matrix(embeddings)
    pairs: list[tuple[str, str, float]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if exclude_same_doc and exclude_same_doc.get(ids[i]) == exclude_same_doc.get(ids[j]):
                continue
            score = float(sims[i, j])
            if score >= threshold:
                pairs.append((ids[i], ids[j], score))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs
