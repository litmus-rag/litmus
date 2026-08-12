"""Step 1.3/8: simple retrieve-then-generate RAG pipeline.

Kept deliberately simple per eval_plan.md §1.3 — we are not benchmarking RAG
quality, just demonstrating litmus's eval framework works. Architecture:
sentence-transformers embedding index over chunks + cosine top-k retrieval +
one LLMClient completion call. Reuses litmus.ingest.embedder (the same
all-MiniLM-L6-v2 model litmus itself uses) rather than a separate embedding
stack, and litmus.llm.client.LLMClient (same conf-loading, same deployment)
rather than a separate LLM call path. One instance is built per corpus's
chunk set and reused across both the litmus- and Ragas-generated eval sets
for that corpus (same RAG system for both, per §0.5).
"""

from __future__ import annotations

import numpy as np

from litmus.ingest.embedder import embed_texts
from litmus.llm.client import LLMClient

from eval_study import config

RAG_PROMPT = """Answer the question using only the information in the provided context. \
If the context does not contain enough information to answer, say so explicitly.

<context>
{context}
</context>

<question>
{question}
</question>

Answer:"""


class SimpleRAG:
    def __init__(self, chunk_texts: list[str], top_k: int = 5, llm_model: str = config.FIXED_LLM):
        self.chunk_texts = chunk_texts
        self.top_k = top_k
        self.client = LLMClient(model=llm_model)
        self.chunk_embeddings = embed_texts(chunk_texts) if chunk_texts else np.zeros((0, 384), dtype=np.float32)

    def retrieve(self, question: str) -> list[str]:
        if len(self.chunk_texts) == 0:
            return []
        q_emb = embed_texts([question])[0]
        sims = self.chunk_embeddings @ q_emb
        top_idx = np.argsort(-sims)[: self.top_k]
        return [self.chunk_texts[i] for i in top_idx]

    def __call__(self, question: str) -> dict:
        contexts = self.retrieve(question)
        prompt = RAG_PROMPT.format(context="\n\n---\n\n".join(contexts), question=question)
        try:
            answer = self.client.complete(prompt, max_tokens=500)
        except Exception as e:  # noqa: BLE001
            answer = ""
        return {"answer": answer, "contexts": contexts}
