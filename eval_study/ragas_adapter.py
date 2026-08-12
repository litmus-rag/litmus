"""Step 7: Ragas TestsetGenerator adapter, wired to the same azure/gpt-5.4
deployment and local sentence-transformers embeddings used everywhere else
in this study (eval_plan.md §0.5 — model consistency across roles).

Normalizes Ragas's Testset output to the plain question/contexts/ground_truth
shape (§3.1's "thin adapter is enough" instruction) rather than forcing it
into litmus's EvalRecord schema.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

import truststore

truststore.inject_into_ssl()

from langchain_core.documents import Document as LCDocument
from openai import AzureOpenAI
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import llm_factory
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from sentence_transformers import SentenceTransformer

from eval_study import config


import threading

_EMBED_LOCK = threading.Lock()


class LocalEmbeddings(BaseRagasEmbeddings):
    """Wraps the same all-MiniLM-L6-v2 model litmus's own embedder uses
    (litmus.ingest.embedder) — no Azure embeddings deployment is confirmed
    live on this resource (see CLAUDE.md).

    Ragas's knowledge-graph transforms fire many concurrent aembed_* calls
    from asyncio + a thread-pool executor. Uncoordinated concurrent calls
    into one SentenceTransformer/libtorch instance from multiple threads
    reliably SIGSEGVs on this machine (confirmed via macOS crash report:
    EXC_BAD_ACCESS inside libtorch_cpu.dylib on an asyncio worker thread).
    A single lock serializes all encode() calls -- embeddings are fast
    enough locally that this isn't a bottleneck.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.set_run_config(RunConfig())

    def embed_query(self, text):
        with _EMBED_LOCK:
            return self.model.encode([text], normalize_embeddings=True).tolist()[0]

    def embed_documents(self, texts):
        with _EMBED_LOCK:
            return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)


def build_ragas_llm(model_name: str = "gpt-5.4"):
    client = AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    llm = llm_factory(model_name, provider="openai", client=client)
    # ragas's reasoning-model auto-detection parses the version out of the
    # model string with int(), which raises on dotted ids like "gpt-5.4" and
    # is silently swallowed -- so max_tokens never gets remapped to
    # max_completion_tokens automatically for this deployment. Same
    # max_tokens/max_completion_tokens quirk CLAUDE.md documents for
    # azure_llm.py's direct calls; bypass by setting model_args directly.
    llm.model_args.pop("max_tokens", None)
    llm.model_args.pop("top_p", None)
    llm.model_args["max_completion_tokens"] = 3000
    llm.model_args["temperature"] = 1.0
    return llm


def load_docs_dir(docs_dir: pathlib.Path) -> list[LCDocument]:
    files = sorted(pathlib.Path(docs_dir).glob("*.txt"))
    return [LCDocument(page_content=f.read_text(), metadata={"filename": f.name}) for f in files]


def generate_ragas_testset(docs_dir: pathlib.Path, testset_size: int, seed: int, max_workers: int = 2):
    llm = build_ragas_llm(config.FIXED_LLM.split("/")[-1])
    embeddings = LocalEmbeddings()
    docs = load_docs_dir(docs_dir)

    gen = TestsetGenerator(llm=llm, embedding_model=embeddings)
    rc = RunConfig(seed=seed, max_workers=max_workers, timeout=300)
    testset = gen.generate_with_langchain_docs(docs, testset_size=testset_size, run_config=rc)
    return testset


def normalize_testset(testset) -> list[dict]:
    """Normalize Ragas's Testset to {question, contexts, ground_truth} rows."""
    df = testset.to_pandas()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "question": row["user_input"],
                "contexts": list(row["reference_contexts"]),
                "ground_truth": row["reference"],
                "synthesizer_name": row.get("synthesizer_name", ""),
            }
        )
    return rows


def run_for_corpus(docs_dir: pathlib.Path, testset_size: int, save_path: pathlib.Path) -> list[dict]:
    print(f"Generating Ragas testset for {docs_dir} (size={testset_size}, seed={config.FIXED_SEED})")
    testset = generate_ragas_testset(docs_dir, testset_size, config.FIXED_SEED)
    rows = normalize_testset(testset)
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} rows to {save_path}")
    return rows


def main() -> None:
    import litmus

    eval_set_a = litmus.load(str(config.CORPUS_A_DIR / "litmus_eval_set.json"))
    eval_set_b = litmus.load(str(config.CORPUS_B_DIR / "litmus_eval_set.json"))

    run_for_corpus(config.CORPUS_A_DOCS_DIR, len(eval_set_a.records), config.CORPUS_A_DIR / "ragas_eval_set.json")
    run_for_corpus(config.CORPUS_B_DOCS_DIR, len(eval_set_b.records), config.CORPUS_B_DIR / "ragas_eval_set.json")


if __name__ == "__main__":
    main()
